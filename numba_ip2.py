import numpy as np

from numba import cuda, float64

from scipy.linalg import cho_factor, cho_solve

from dipy.reconst.mcsd import MSDeconvFit

# GPU implementation of primal-dual Interior Point method
# for convex quadratic constrained optimization (QP)
# Specifically, ||Rx-d||_2 where Ax>=b
# parallelized to solve 10s-100s of thousands of QPs
# Given same R, A, b but different d (doesn't use this fact right now though)
# ultimately used to fit MSMT CSD

USE_DEBUGGER = False

@cuda.jit()
def parallel_qp_fit(Rt, R_pinv, Q, A, b, x0, data, results):
    '''
    Solves 1/2*x^t*Q*x+(Rt*d)^t*x given Ax>=0
    '''
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    k = cuda.blockIdx.z

    if max(data[i, j, k]) == 0:
        for l in block_range(results.shape[3]):
            results[i, j, k, l] = 0.0
        cuda.syncthreads()
    else:
        interior_point(
            Rt, R_pinv, Q, A, b, x0, data[i, j, k], results[i, j, k])


@cuda.jit(device=True)
def interior_point(Rt, R_pinv, G, A, b, x0, d, x):
    cp=0.9
    mu=1.0
    max_iter=1000
    tol=1e-6
    tau=0.95

    m, n = A.shape
    nm = n+m

    shared_memory = cuda.shared.array(0, dtype=float64)
    c = shared_memory[:n]
    y = shared_memory[n:n+m]
    l = shared_memory[n+m:n+2*m]
    dx = shared_memory[n+2*m:2*n+2*m]
    dy = shared_memory[2*n+2*m:2*n+3*m]
    dl = shared_memory[2*n+3*m:2*n+4*m]
    rhs1 = shared_memory[2*n+4*m:3*n+4*m]
    rhs2 = shared_memory[3*n+4*m:3*n+5*m]
    Z = shared_memory[3*n+5*m:3*n+6*m]
    schur = shared_memory[3*n+6*m:3*n+6*m+n*n]

    # fun inverse helpers
    pt = 3*n+6*m+n*n
    isz = n
    cgr  = shared_memory[pt      :pt+isz]
    cgp  = shared_memory[pt+isz  :pt+2*isz]
    cgAp = shared_memory[pt+2*isz:pt+3*isz]

    # first check if naive R^+d happens to satisfy the constraints
    cuda.syncthreads()
    for ii in block_range(R_pinv.shape[0]):
        x[ii] = 0.0
        for jj in range(R_pinv.shape[1]):
            x[ii] += R_pinv[ii, jj] * d[jj]
    cuda.syncthreads()

    fails_ineq = 0
    for ii in block_range(A.shape[0]):
        __tmp = 0.0
        for jj in range(A.shape[1]):
            __tmp += A[ii, jj] * x[jj]
        if (__tmp - b[ii]) < -tol:
            fails_ineq = 1
            break

    if not USE_DEBUGGER:
        efi = cuda.ballot_sync(-1, fails_ineq)
        fails_ineq = efi
    cuda.syncthreads()
    if fails_ineq == 0:
        return

    # now calc c
    for ii in block_range(Rt.shape[0]):
        c[ii] = 0.0
        for jj in range(Rt.shape[1]):
            c[ii] += Rt[ii, jj] * d[jj]
    for ii in block_range(n):
        x[ii] = x0[ii]
    for ii in block_range(m):
        y[ii] = 1.0
    for ii in block_range(m):
        l[ii] = 1.0
    cuda.syncthreads()

    for __iter in range(max_iter):
        if __iter != 0:
            mu *= cp

        # z = l \ y
        for i in block_range(m):
            if abs(y[i]) < tol:
                if y[i] < 0:
                    Z[i]=-tol
                else:
                    Z[i]=tol
            else:
                Z[i] = l[i] / y[i]
        cuda.syncthreads()

        # rhs[n:n+m] = -(A.dot(x0) - y0 - b) + (-y + sigma*mu/l)
        for i in block_range(m):
            A_dot_x_i = 0.0
            for j in range(n):
                A_dot_x_i += A[i, j] * x[j]
            
            rp_i = A_dot_x_i - y[i] - b[i]
            rhs2[i] = -rp_i + (-y[i] + mu/l[i])
            dy[i] = rp_i
        cuda.syncthreads()

        # rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
        for i in block_range(n):
            G_dot_x_i = 0.0
            for j in range(n):
                G_dot_x_i += G[i, j] * x[j]

            A_T_dot_l_i = 0.0
            for j in range(m):
                A_T_dot_l_i += A[j, i] * l[j]

            rhs1[i] = -(G_dot_x_i - A_T_dot_l_i + c[i])

            # rhs1 + A.T@(z*b)
            for j in range(m):
                rhs1[i] += A[j, i]*rhs2[j]*Z[j]
        cuda.syncthreads()

        # dx = np.linalg.solve(G+A.T@np.diag(Z)@A, rhs1)
        for i in block_range(n):
            for j in range(n):
                schur[i*n+j] = G[i, j]
                for k in range(m):
                    schur[i*n+j] += A[k, i] * A[k, j] * Z[k]
        cuda.syncthreads()

        solved_inverse = cg(
            schur, rhs1, dx,
            cgr, cgp, cgAp,
            n, tol)
        cuda.syncthreads()
        if solved_inverse == 0:
            for ii in block_range(n):
                x[ii] = 0
            cuda.syncthreads()
            return

        # dl = z*(rhs2-A@dx)
        for i in block_range(m):
            dl[i] = rhs2[i]
            for j in range(n):
                dl[i] -= A[i, j]*dx[j]
            dl[i] *= Z[i]
        cuda.syncthreads()

        # dy = A@dx+(A.dot(x0) - y0 - b)
        for i in block_range(m):
            for j in range(n):
                dy[i] += A[i, j]*dx[j]
        cuda.syncthreads()

        # calculate step size
        beta = 1.0
        sigma = 1.0
        for ii in block_range(m):
            if dy[ii] < 0:
                sigma = min(sigma, -y[ii]/dy[ii])
            if dl[ii] < 0:
                beta = min(beta, -l[ii]/dl[ii])
        beta = min(1.0, tau*beta)
        sigma = min(1.0, tau*sigma)
        alpha = min(beta, sigma)
        if not USE_DEBUGGER:
            ii = 1
            while ii < cuda.blockDim.x:
                oa = cuda.shfl_xor_sync(-1, alpha, ii)
                alpha = min(alpha, oa)
                ii = ii*2
        cuda.syncthreads()
        # print(np.asarray(x))
        # print(alpha)
        # print(v)

        # time to step
        tdx = 0
        tdy = 0
        tdl = 0
        for ii in block_range(n):
            x[ii] += alpha*dx[ii]
            tdx += abs(dx[ii])
        for ii in block_range(m):
            y[ii] += alpha*dy[ii]
            tdy += abs(dy[ii])
            l[ii] += alpha*dl[ii]
            tdl += abs(dl[ii])
        if not USE_DEBUGGER:
            ii = 1
            while ii < cuda.blockDim.x:
                ot = cuda.shfl_xor_sync(-1, tdx, ii)
                tdx += ot
                ot = cuda.shfl_xor_sync(-1, tdy, ii)
                tdy += ot
                ot = cuda.shfl_xor_sync(-1, tdl, ii)
                tdl += ot
                ii = ii*2
        cuda.syncthreads()

        if (alpha*tdx < n*tol) and (alpha*tdy < m*tol) and (alpha*tdl < m*tol):
            return

    # failed to solve
    for ii in block_range(n):
        x[ii] = 0
    cuda.syncthreads()

@cuda.jit(device=True)
def cg(A, b, x,
       r, p, Ap,
       n, tol):
    max_iter=1000

    # r = b - np.dot(A, x)
    for ii in block_range(n):
        r[ii] = b[ii]
        for jj in range(n):
            r[ii] -= A[ii*n+jj]*x[jj]
    cuda.syncthreads()
    
    # p = r.copy()
    for ii in block_range(n):
        p[ii] = r[ii]
    cuda.syncthreads()
    
    #rs_old = np.dot(r.T, r)
    rs_old = parallel_dot(r, r)

    for _ in range(max_iter):
        # Ap = np.dot(A, p)
        for ii in block_range(n):
            Ap[ii] = 0
            for jj in range(n):
                Ap[ii] += A[ii*n+jj]*p[jj]
        cuda.syncthreads()

        # alpha = rs_old / np.dot(p.T, Ap)
        alpha = parallel_dot(p, Ap)
        alpha = rs_old / alpha

        # x = x + alpha * p
        # r = r - alpha * Ap
        for ii in block_range(n):
            x[ii] += alpha*p[ii]
            r[ii] -= alpha*Ap[ii]
        cuda.syncthreads()

        rs_new = parallel_dot(r, r)
        
        if rs_new < tol:
            return 1
        
        # p = r + (rs_new / rs_old) * p
        for ii in block_range(n):
            p[ii] = r[ii] + (rs_new / rs_old) * p[ii]
        cuda.syncthreads()

        rs_old = rs_new

    return rs_new < tol**0.5 # close enough

@cuda.jit(device=True)
def parallel_dot(x, y):
    rv = 0
    for ii in block_range(x.shape[0]):
        rv += x[ii]*y[ii]
    if not USE_DEBUGGER:
        ii = 1
        while ii < cuda.blockDim.x:
            os = cuda.shfl_xor_sync(-1, rv, ii)
            rv += os
            ii = ii*2
    cuda.syncthreads()
    return rv

@cuda.jit(device=True)
def block_range(__stop):
    '''
    Assumes blocks are of shape (x, 1, 1)
    '''
    return range(cuda.threadIdx.x, __stop, cuda.blockDim.x)


def fit(self, data):
    coeff = np.zeros((*data.shape[:3], self.fitter._X.shape[1]))

    R = self.fitter._X
    A = self.fitter._reg

    Q = R.T @ R
    x0 = np.linalg.pinv(A) @ np.ones(A.shape[0])
    if np.sum(A@x0 < 0) != 0:
        x0 = np.zeros(A.shape[0])
    
    Rt = cuda.to_device(-R.T)
    R_pinv = cuda.to_device(np.linalg.pinv(R))
    Q = cuda.to_device(Q)
    A = cuda.to_device(A)
    b = cuda.to_device(np.zeros(A.shape[0]))
    x0 = cuda.to_device(x0)

    data = cuda.to_device(data)
    coeff = cuda.to_device(coeff)
    
    m, n = A.shape
    sh_mem = 8*(3*n+6*m+n*n+3*n)

    parallel_qp_fit[
        data.shape[:3], 32,
        0, sh_mem](
            Rt, R_pinv, Q, A, b, x0, data, coeff)

    cuda.current_context().synchronize()
    coeff = coeff.copy_to_host()
    
    print(np.sum(coeff))
    # print(coeff)
    return MSDeconvFit(self, coeff, None)