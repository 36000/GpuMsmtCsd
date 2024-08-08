from numba_ip2 import get_deriv_mat, parallel_qp_fit
import numpy as np
from numba import cuda

def test_ip2():
    Q = np.asarray([[1.0, -1.0], [-1.0, 2.0]])
    Rt = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    d = np.asarray([[[[-2.0, -6.0]]]])
    res = np.zeros((1, 1, 1, Q.shape[1]))
    A = np.asarray([
        [-1.0, -1.0],
        [1.0, -2.0],
        [-2.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0]])
    b = np.asarray([-2.0, -2.0, -3.0, 0.0, 0.0])
    x0 = np.asarray([0.5, 0.5])

    N, y0, l0 = get_deriv_mat(A, Q, x0)

    Rt = cuda.to_device(Rt)
    Q = cuda.to_device(Q)
    A = cuda.to_device(A)
    b = cuda.to_device(b)
    N = cuda.to_device(N)
    x0 = cuda.to_device(x0)
    y0 = cuda.to_device(y0)
    l0 = cuda.to_device(l0)

    d = cuda.to_device(d)
    res = cuda.to_device(res)
    
    m, n = A.shape
    nm2 = n+2*m
    shmem_num_elements = n+2*m+8*nm2
    shmem_sz = shmem_num_elements*4

    parallel_qp_fit[
        (1, 1, 1), 32,
        0, shmem_sz](
            Rt, Q, A, b, N, x0, y0, l0, d, res)

    cuda.current_context().synchronize()
    res = res.copy_to_host()
    
    # print(np.sum(coeff))
    print(res)


if __name__ == "__main__":
    test_ip2()
