import numpy as np
from numba import cuda, float64, float32

from dipy.reconst.mcsd import MSDeconvFit

# GPU implementation of constrained least squares fit (1/2||Rx-d||_2, Ax>=b)
# parallelized to solve 10s-100s of thousands of individual fits
# Given same R, A, b but different d
# (modified to only store adjustments to matrices)
# ultimately used to fit MSMT CSD

USE_DEBUGGER = False

@cuda.jit()
def batch_constrained_ls_fit(Rt, R_pinv, N_inv, A, b, data, results):
    '''
    Solves 1/2||Rx-d||_2, Ax>=b
    '''
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    k = cuda.blockIdx.z

    if max(data[i, j, k]) == 0:
        for l in block_range(results.shape[3]):
            results[i, j, k, l] = 0.0
        cuda.syncthreads()
    else:
        constrained_ls(
            Rt, R_pinv, N_inv, A, b, data[i, j, k], results[i, j, k])


# TODO: some of the shfl_sync's are sums/mins which could be shfl_xor
@cuda.jit(device=True)
def constrained_ls(Rt, R_pinv, N_inv, A, b, d, x):
    tidx = cuda.threadIdx.x
    m, n = A.shape
    tol = 1e-10

    shared_memory = cuda.shared.array(0, dtype=float64)
    c = shared_memory[:n]

    # try R_pinv @ d, and if it satisfies A @ (R_pinv @ d) >= b, return x = R_pinv @ d
    cuda.syncthreads()
    for ii in block_range(R_pinv.shape[0]):
        x[ii] = 0.0
        for jj in range(R_pinv.shape[1]):
            x[ii] += R_pinv[ii, jj] * d[jj]
    cuda.syncthreads()

    fails_ineq = 0
    for ii in range(A.shape[0]):
        __tmp = 0.0
        for jj in range(A.shape[1]):
            __tmp += A[ii, jj] * x[jj]
        if (__tmp - b[ii]) < -tol:
            fails_ineq = 1
            break

    # if not USE_DEBUGGER: # TODO
    #     everyone_fails_ineq = cuda.ballot_sync(-1, fails_ineq)
    #     fails_ineq = everyone_fails_ineq
    cuda.syncthreads()
    if fails_ineq == 0:
        return

    # calculate c=Rtd
    for ii in block_range(Rt.shape[0]):
        c[ii] = 0.0
        for jj in range(Rt.shape[1]):
            c[ii] += Rt[ii, jj] * d[jj]
    cuda.syncthreads()

    # [x y] = N_inv@[c b], but only calc x
    for ii in block_range(n):
        x[ii] = 0
        for jj in range(n):
            x[ii] += N_inv[ii, jj] * c[jj]
        for jj in range(m):
            x[ii] += N_inv[ii, n+jj] * b[jj]
    cuda.syncthreads()


@cuda.jit(device=True)
def block_range(__stop):
    '''
    Assumes blocks are of shape (x, 1, 1)
    '''
    return range(cuda.threadIdx.x, __stop, cuda.blockDim.x)


def prep_problem(R, A, epsilon=1e-3):
    m, n = A.shape

    # if np.linalg.cond(R) > 1/epsilon: # TODO: some regularization needed; CSF high AP
    #     for ii in range(R.shape[0]):
    #         norm = np.linalg.norm(R[ii])
    #         if norm < 1:
    #             norm = 1
    #         R[ii] /= norm

    N = np.zeros((n+m, n+m))
    N[:n, :n] = R.T @ R
    N[:n, n:] = A.T
    N[n:, :n] = A

    U, S, VT = np.linalg.svd(N)
    S_inv = np.array([1/s if s > epsilon else 0 for s in S])
    N_inv_svd = VT.T @ np.diag(S_inv) @ U.T

    U, S, VT = np.linalg.svd(R, full_matrices=False)
    S_inv = np.array([1/s if s > epsilon else 0 for s in S])
    R_pinv_svd = VT.T @ np.diag(S_inv) @ U.T

    return R.T, R_pinv_svd, N_inv_svd


def fit(self, data):
    coeff = np.zeros((*data.shape[:3], self.fitter._X.shape[1]))

    R = self.fitter._X
    A = self.fitter._reg
    b = np.zeros(A.shape[0])

    Rt, R_pinv, N_inv = prep_problem(R, A)

    Rt = cuda.to_device(Rt)
    R_pinv = cuda.to_device(R_pinv)
    N_inv = cuda.to_device(N_inv)
    A = cuda.to_device(A)
    b = cuda.to_device(b)

    data = cuda.to_device(data)
    coeff = cuda.to_device(coeff)

    m, n = A.shape
    shmem_sz = n*8

    batch_constrained_ls_fit[
        data.shape[:3], 32,
        0, shmem_sz](
            Rt, R_pinv, N_inv, A, b, data, coeff)

    cuda.current_context().synchronize()
    coeff = coeff.copy_to_host()

    return MSDeconvFit(self, coeff, None)
