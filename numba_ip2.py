import numpy as np
from numba import cuda, float64, float32

from dipy.reconst.mcsd import MSDeconvFit

# GPU implementation of primal-dual Interior Point method
# for convex quadratic constrained optimization (QP)
# parallelized to solve 10s-100s of thousands of QPs
# Given same Q, A, b but different c
# (modified to only store adjustments to matrices)
# ultimately used to fit MSMT CSD
# https://www.maths.ed.ac.uk/~gondzio/reports/ipmXXV.pdf

USE_DEBUGGER = False

@cuda.jit()
def parallel_qp_fit(Rt, Q, A, b, A_invb, AT_inv, x0, y0, l0, data, results):
    '''
    Solves 1/2*x^t*Q*x+(Rt*d)^t*x given Ax>=0
    '''
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    k = cuda.blockIdx.z

    if max(data[i, j, k]) == 0:
        for l in block_range(data.shape[3]):
            results[i, j, k, l] = 0.0
        cuda.syncthreads()
    else:
        interior_point(
            Rt, Q, A, b, A_invb, AT_inv, x0, y0, l0, data[i, j, k], results[i, j, k])


# TODO: some of the shfl_sync's are sums/mins which could be shfl_xor
# TODO: this works perfectly for Ax=b, x>=0. but it needs to be modified for Ax>=b.
@cuda.jit(device=True)
def interior_point(Rt, Q, A, b, A_invb, AT_inv, x0, y0, z0, d, x):
    alpha0 = 0.99
    cp = 0.9 # TODO: this can be (0, 1). but what should it be?????
    max_iter=500
    tol=1e-6

    tidx = cuda.threadIdx.x
    m, n = A.shape
    nm = n+m

    shared_memory = cuda.shared.array(0, dtype=float64)
    c = shared_memory[:n]
    y = shared_memory[n:n+m]
    z = shared_memory[n+m:2*n+m]
    dx = shared_memory[2*n+m:3*n+m]
    dy = shared_memory[3*n+m:3*n+2*m]
    dz = shared_memory[3*n+2*m:4*n+2*m]

    cuda.syncthreads()
    for ii in block_range(Rt.shape[0]):
        c[ii] = 0.0
        for jj in range(Rt.shape[1]):
            c[ii] += Rt[ii, jj] * d[jj]

    for ii in block_range(n):
        x[ii] = x0[ii]
    for ii in block_range(m):
        y[ii] = y0[ii]
    for ii in block_range(n):
        z[ii] = z0[ii]
    cuda.syncthreads()

    v = 0.0
    if tidx == 0:
        for i in range(n):
            v += x[i] * z[i]
        v /= n
    if not USE_DEBUGGER:
        v = cuda.shfl_sync(-1, v, 0)
    cuda.syncthreads()
    for __iter in range(max_iter):
        v*=cp

        for ii in block_range(n):
            dx[ii] = A_invb[ii] - x[ii]
        cuda.syncthreads()

        # dy = np.linalg.pinv(A.T) @ (-Q-ThetaInv @ dx + (c - A.T@y + Q@x - (cp*v)/x))
        for ii in block_range(m):
            dy[ii] = 0
            for jj in range(n):
                Qtheta_dx_j = 0
                for kk in range(n):
                    Qtheta_dx_j += Q[jj, kk] * dx[kk] # Q part
                Qtheta_dx_j += safe_divide(z[jj], x[jj]) * dx[jj] # Theta adjustment
            
                intermediate_result_j = c[jj]
                for kk in range(m):
                    intermediate_result_j -= A[kk, jj] * y[kk]
                for kk in range(n):
                    intermediate_result_j += Q[jj, kk] * x[kk]
                intermediate_result_j -= safe_divide(cp*v, x[jj])

                dy[ii] += AT_inv[ii, jj] * (Qtheta_dx_j + intermediate_result_j)

        for ii in block_range(n):
            dz[ii] = safe_divide(cp*v-z[ii]*dx[ii], x[ii])-z[ii]
        cuda.syncthreads()

        # calculate step size
        beta = 1.0
        sigma = 1.0
        if tidx == 0:
            for ii in range(n):
                if dx[ii] < 0:
                    sigma = max(sigma, -x[ii]/dx[ii])
                if dz[ii] < 0:
                    beta = max(beta, -z[ii]/dz[ii])
            beta = min(1.0, alpha0*beta)
            sigma = min(1.0, alpha0*sigma)
        if not USE_DEBUGGER:
            beta = cuda.shfl_sync(-1, beta, 0)
            sigma = cuda.shfl_sync(-1, sigma, 0)
        else:
            beta = 0.05
            sigma = 0.05
        cuda.syncthreads()

        # print(np.asarray(x))
        # print(v)

        # time to step
        for ii in block_range(n):
            z[ii] += beta*dz[ii]
            x[ii] += sigma*dx[ii]
        for ii in block_range(m):
            y[ii] += beta*dy[ii]
        cuda.syncthreads()

        if norm_sq(dx) < tol:
            break


@cuda.jit(device=True)
def safe_divide(n, d):
    tol = 1e-6
    if d < 0:
        if -d < tol:
            d = -tol
    else:
        if d < tol:
            d = tol
    return n/d

@cuda.jit(device=True)
def norm_sq(x):
    __sum = 0
    if cuda.threadIdx.x == 0:
        for ii in range(x.shape[0]):
            __sum += x[ii]*x[ii]
    if not USE_DEBUGGER:
        __sum = cuda.shfl_sync(-1, __sum, 0)
    cuda.syncthreads()
    return __sum

@cuda.jit(device=True)
def block_range(__stop):
    '''
    Assumes blocks are of shape (x, 1, 1)
    '''
    return range(cuda.threadIdx.x, __stop, cuda.blockDim.x)

def prep_problem(A, b):
    m, n = A.shape

    y0 = np.zeros(m)
    z0 = np.ones(m)

    A_invb = np.linalg.pinv(A) @ b
    AT_inv = np.linalg.pinv(A.T)

    return A_invb, AT_inv, y0, z0


def fit(self, data):
    coeff = np.zeros((*data.shape[:3], self.fitter._X.shape[1]))

    R = self.fitter._X
    Q = R.T @ R
    A = -pself.fitter._reg # TODO: should be negative?
    x0 = np.linalg.pinv(A) @ np.ones(A.shape[0])
    b = np.zeros(A.shape[0])

    A_invb, AT_inv, y0, z0 = prep_problem(A, b)

    Rt = cuda.to_device(-R.T) # TODO: should be negative?
    Q = cuda.to_device(Q)
    A = cuda.to_device(A)
    b = cuda.to_device(b)
    A_invb = cuda.to_device(A_invb)
    AT_inv = cuda.to_device(AT_inv)
    x0 = cuda.to_device(x0)
    y0 = cuda.to_device(y0)
    z0 = cuda.to_device(z0)

    data = cuda.to_device(data)
    coeff = cuda.to_device(coeff)

    m, n = A.shape
    shmem_num_elements = 4*n+2*m
    shmem_sz = shmem_num_elements*8
    print(shmem_sz)

    parallel_qp_fit[
        data.shape[:3], 32,
        0, shmem_sz](
            Rt, Q, A, b, A_invb, AT_inv, x0, y0, z0, data, coeff)

    cuda.current_context().synchronize()
    coeff = coeff.copy_to_host()

    print(coeff)
    print(np.sum(coeff))
    return MSDeconvFit(self, coeff, None)
