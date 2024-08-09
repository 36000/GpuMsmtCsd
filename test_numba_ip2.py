from numba_ip2 import prep_problem, parallel_qp_fit, USE_DEBUGGER
import numpy as np
from numba import cuda

def test_ip2():
    Rt = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    if False: # 0.688 1.188
        Q = np.asarray([[1.0, -1.0], [-1.0, 2.0]])
        d = np.asarray([[[[-2.0, -6.0]]]])
        A = np.asarray([
            [-1.0, -1.0],
            [1.0, -2.0],
            [-2.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0]])
        b = np.asarray([-2.0, -2.0, -3.0, 0.0, 0.0])
        x0 = np.asarray([0.5, 0.5])
    else: # 3.114 3.295 2.295 1.296
        Q = np.array([
            [3.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0],
            [0.0, -1.0, 5.0, -1.0],
            [0.0, 0.0, -1.0, 6.0]
        ])
        d = np.array([[[[-8.0, -16.0, -4.0, -6.0]]]])
        A = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [-1.0, 2.0, -1.0, 2.0]
        ])
        b = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 1.0, 1.0, 1.0, 5.0])

        # Initial guess for the variables
        x0 = np.zeros(Q.shape[0]) #np.linalg.pinv(A) @ np.ones(A.shape[0]) #np.zeros(Q.shape[0])

    res = np.zeros((1, 1, 1, Q.shape[1]))
    A_invb, AT_inv, y0, z0 = prep_problem(A, b)

    Rt = cuda.to_device(Rt)
    Q = cuda.to_device(Q)
    A = cuda.to_device(A)
    b = cuda.to_device(b)
    A_invb = cuda.to_device(A_invb)
    AT_inv = cuda.to_device(AT_inv)
    x0 = cuda.to_device(x0)
    y0 = cuda.to_device(y0)
    z0 = cuda.to_device(z0)

    d = cuda.to_device(d)
    res = cuda.to_device(res)
    
    m, n = A.shape
    nm = n+m
    shmem_num_elements = 2*n+m+8*nm
    shmem_sz = shmem_num_elements*8

    if USE_DEBUGGER:
        threads_per_block = 1
    else:
        threads_per_block = 32

    parallel_qp_fit[
        (1, 1, 1), threads_per_block,
        0, shmem_sz](
            Rt, Q, A, b, A_invb, AT_inv, x0, y0, z0, d, res)

    print(res)
    cuda.current_context().synchronize()
    res = res.copy_to_host()
    
    # print(np.sum(coeff))
    print(res)


if __name__ == "__main__":
    test_ip2()
