import numpy as np
from numba import cuda, float64, float32

from dipy.reconst.mcsd import MSDeconvFit

# GPU implementation of primal-dual Interior Point method
# for convex quadratic constrained optimization (QP)
# parallelized to solve 10s-100s of thousands of QPs
# Given same Q, A, b but different c
# ultimately used to fit MSMT CSD
# Based on: https://acme.byu.edu/0000017a-1bb8-db63-a97e-7bfa0bd60001/vol2lab21interiorpoint2-pdf 
# With modification in finding inverse of the derivative matrix, using BiCGSTAB
# and only storing adjustments to the matrix

@cuda.jit()
def parallel_qp_fit(Rt, Q, A, b, N0, x0, y0, l0, data, results):
    '''
    Solves 1/2*x^t*Q*x+(Rt*d)^t*x given Ax>=0
    '''
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    k = cuda.blockIdx.z

    if max(data[i, j, k]) == 0:
        for l in range(data.shape[3]):
            data[i, j, k, l] = 0.0
    else:
        interior_point(
            Rt, Q, A, b, N0, x0, y0, l0, data[i, j, k], results[i, j, k])
    # print(str(i) + str(j) + str(k) + str(np.asarray(results[i, j, k])))


# TODO: some of the shfl_sync's are sums which could be shfl_xor
@cuda.jit(device=True)
def interior_point(Rt, G, A, b, N0, x0, y0, l0, d, x):
    center_parameter_init=0.0
    center_parameter_stepping=0.1
    max_iter=21
    tol=1e-6
    tau=0.95
    tau_inc=(1.0-tau)/max_iter # tau approaches 1

    tidx = cuda.threadIdx.x
    m, n = A.shape
    nm2 = n+2*m

    shared_memory = cuda.shared.array(0, dtype=float32) # TODO: bring this up to float64
    c = shared_memory[:n]
    y = shared_memory[n:n+m]
    l = shared_memory[n+m:n+2*m]
    dxyl = shared_memory[n+2*m:n+2*m+nm2]
    rhs = shared_memory[n+2*m+nm2:n+2*m+2*nm2]

    # fun inverse helpers
    r =     shared_memory[n+2*m+2*nm2:n+2*m+3*nm2]
    r_hat = shared_memory[n+2*m+3*nm2:n+2*m+4*nm2]
    v_bic = shared_memory[n+2*m+4*nm2:n+2*m+5*nm2]
    p_bic = shared_memory[n+2*m+5*nm2:n+2*m+6*nm2]
    s_bic = shared_memory[n+2*m+6*nm2:n+2*m+7*nm2]
    t_bic = shared_memory[n+2*m+7*nm2:n+2*m+8*nm2]

    cuda.syncthreads()
    for ii in block_range(Rt.shape[0]):
        c[ii] = 0.0
        for jj in range(Rt.shape[1]):
            c[ii] += Rt[ii, jj] * d[jj]

    for ii in block_range(n):
        x[ii] = x0[ii]
    for ii in block_range(m):
        y[ii] = y0[ii]
    for ii in block_range(m):
        l[ii] = l0[ii]
    cuda.syncthreads()

    center_parameter = center_parameter_init
    for __iter in range(max_iter):
        # print("here")
        # rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
        for i in block_range(n):
            G_dot_x_i = 0.0
            for j in range(n):
                G_dot_x_i += G[i, j] * x[j]

            A_T_dot_l_i = 0.0
            for j in range(m):
                A_T_dot_l_i += A[j, i] * l[j]

            rhs[i] = -(G_dot_x_i - A_T_dot_l_i + c[i])

        # rhs[n:n+m] = -(A.dot(x0) - y0 - b)
        for i in block_range(m):
            A_dot_x_i = 0.0
            for j in range(n):
                A_dot_x_i += A[i, j] * x[j]
            
            rhs[n + i] = -(A_dot_x_i - y[i] - b[i])

        # rhs[n+m:] = -(y0*l0)+v*sigma
        for i in block_range(m):
            rhs[n + m + i] = -(y[i] * l[i])
        cuda.syncthreads()

        if tidx == 0:
            v = 0.0
            for i in range(m):
                v += y[i] * l[i]
            v /= m
        v = cuda.shfl_sync(-1, v, 0)

        for i in block_range(m):
            rhs[n + m + i] += v*center_parameter
        cuda.syncthreads()

        # sol = la.solve(N, rhs)
        # dx = sol[:n]
        # dy = sol[n:n+m]
        # dl = sol[n+m:]
        # N = np.zeros((n+m+m, n+m+m))
        solved_inverse = bicgstab(
            N0, rhs, dxyl,
            r, r_hat, v_bic, p_bic, s_bic, t_bic,
            y, l, n, m, nm2)

        center_parameter = center_parameter_stepping
        if __iter == 0:
            # y0 = np.maximum(1, np.abs(y0 + dy))
            # l0 = np.maximum(1, np.abs(l0+dl))
            for ii in block_range(m):
                y[ii] = max(1, abs(y[ii] + dxyl[n+ii]))
                l[ii] = max(1, abs(l[ii] + dxyl[n+m+ii]))
                dxyl[n+ii] = y[ii] - y0[ii]
                dxyl[n+m+ii] = l[ii] - l0[ii]
            for ii in block_range(n):
                dxyl[ii] = 0
            cuda.syncthreads()
        else:
            # calculate step size
            alpha = 0
            if tidx == 0:
                beta = 1.0
                sigma = 1.0
                for ii in range(m):
                    if dxyl[n+ii] < 0:
                        sigma = min(sigma, -y[ii]/dxyl[n+ii])
                    if dxyl[n+m+ii] < 0:
                        beta = min(beta, -l[ii]/dxyl[n+m+ii])
                beta = min(1.0, tau*beta)
                sigma = min(1.0, tau*sigma)
                alpha = min(beta, sigma)
            alpha = cuda.shfl_sync(-1, alpha, 0)
            tau += tau_inc
            
            # print(np.asarray(x))
            # print(alpha)
            # print(v)

            # time to step
            for ii in block_range(n):
                x[ii] += alpha*dxyl[ii]
            for ii in block_range(m):
                y[ii] += alpha*dxyl[n+ii]
                l[ii] += alpha*dxyl[n+m+ii]
            cuda.syncthreads()

            if (v < tol) or (alpha < tol):
                break

@cuda.jit(device=True)
def bicgstab(A, b, x, r, r_hat, v, p, s, t, y, l, n, m, nm2):
    tol=1e-10
    max_iter=100

    tidx = cuda.threadIdx.x

    matmul_A(A, x, r, y, l, n, m)
    cuda.syncthreads()
    for ii in block_range(nm2):
        r[ii] = b[ii] - r[ii]
        r_hat[ii] = r[ii]
        v[ii] = 0
        p[ii] = 0
    cuda.syncthreads()

    rho_old = alpha = omega = 1.0
    
    for iter_count in range(max_iter):
        rho_new = 0.0
        if tidx == 0:
            for ii in range(nm2):
                rho_new += r_hat[ii]*r[ii]
        rho_new = cuda.shfl_sync(-1, rho_new, 0)
        cuda.syncthreads()

        if rho_new == 0:
            return False

        if iter_count == 0:
            for ii in block_range(nm2):
                p[ii] = r[ii]
        else:
            beta = (rho_new / rho_old) * (alpha / omega)
            for ii in block_range(nm2):
                p[ii] = r[ii] + beta * (p[ii] - omega * v[ii])
        cuda.syncthreads()

        matmul_A(A, p, v, y, l, n, m)
        cuda.syncthreads()

        alpha = 0.0
        if tidx == 0:
            for ii in range(nm2):
                alpha += r_hat[ii]*v[ii]
            alpha = rho_new / alpha
        alpha = cuda.shfl_sync(-1, alpha, 0)
        cuda.syncthreads()

        for ii in block_range(nm2):
            s[ii] = r[ii] - alpha * v[ii]
        cuda.syncthreads()

        if norm_sq(s) < tol:
            for ii in block_range(nm2):
                x[ii] += alpha * p[ii]
            cuda.syncthreads()
            return True

        matmul_A(A, s, t, y, l, n, m)
        cuda.syncthreads()

        omega = 0
        if tidx == 0:
            omg_helper = 0
            for ii in range(nm2):
                omega += t[ii]*s[ii]
                omg_helper += t[ii]*t[ii]
            omega = omega/omg_helper
        omega = cuda.shfl_sync(-1, omega, 0)
        cuda.syncthreads()

        for ii in block_range(nm2):
            x[ii] += alpha*p[ii] + omega*s[ii]
            r[ii] = s[ii] - omega * t[ii]
        cuda.syncthreads()

        if norm_sq(r) < tol:
            return True

        if omega == 0:
            return False

        rho_old = rho_new

    return True

@cuda.jit(device=True)
def matmul_A(A, b, res, y, l, n, m):
    for ii in block_range(n+m):
        res[ii] = 0
        for jj in range(n+2*m):
            res[ii] += A[ii, jj]*b[jj]
    for ii in block_range(m):
        res[n+m+ii] = l[ii]*b[ii+n] + y[ii]*b[ii+n+m]

@cuda.jit(device=True)
def norm_sq(x):
    __sum = 0
    if cuda.threadIdx.x == 0:
        for ii in range(x.shape[0]):
            __sum += x[ii]*x[ii]
    __sum = cuda.shfl_sync(-1, __sum, 0)
    cuda.syncthreads()
    return __sum

@cuda.jit(device=True)
def block_range(__stop):
    '''
    Assumes blocks are of shape (x, 1, 1)
    '''
    return range(cuda.threadIdx.x, __stop, cuda.blockDim.x)

def get_deriv_mat(A, Q, x0):
    m, n = A.shape
    
    y0 = np.ones(m)
    l0 = np.ones(m)

    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = Q
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)

    return N, y0, l0


def fit(self, data):
    coeff = np.zeros((*data.shape[:3], self.fitter._X.shape[1]))

    R = self.fitter._X
    Q = R.T @ R
    A = self.fitter._reg
    x0 = np.linalg.pinv(A) @ np.ones(A.shape[0])

    N, y0, l0 = get_deriv_mat(A, Q, x0)

    Rt = cuda.to_device(-R.T) # TODO: should be negative?
    Q = cuda.to_device(Q)
    A = cuda.to_device(A)
    b = cuda.to_device(np.zeros(A.shape[0]))
    N = cuda.to_device(N)
    x0 = cuda.to_device(x0)
    y0 = cuda.to_device(y0)
    l0 = cuda.to_device(l0)

    data = cuda.to_device(data)
    coeff = cuda.to_device(coeff)

    m, n = A.shape
    nm2 = n+2*m
    shmem_num_elements = n+2*m+8*nm2
    shmem_sz = shmem_num_elements*4
    print(shmem_sz)

    parallel_qp_fit[
        # data.shape[:3], 1,
        (data.shape[0]//4, data.shape[1]//4, data.shape[2]//4), 1,
        32, shmem_sz](
            Rt, Q, A, b, N, x0, y0, l0, data, coeff)

    cuda.current_context().synchronize()
    coeff = coeff.copy_to_host()
    
    print(np.sum(coeff))
    # print(coeff)
    return MSDeconvFit(self, coeff, None)
