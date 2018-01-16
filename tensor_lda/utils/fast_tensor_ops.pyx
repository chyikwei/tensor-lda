cimport cython
cimport numpy as np
import numpy as np

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def tensor_3d_permute_231(np.ndarray[ndim=2, dtype=np.float64_t, mode="c"] tensor):
    """Permute 3-D tensor with 2-3-1 Mode

    This is a fast version to calculate mode (2,3,1) permutaion of
    a 3-D tensor.

    Parameters
    ----------
    tensor :  array, (n, n * n)
        3D tensor in unfolded format.

    tensor_shape : integer triple
        Shape of the tensor. Since tensor is in
        unfolded format. We need it's real format
        to calculate permutation.

    Returns
    -------
    permuted_tensor: array, (n, n * n)
        Tensor after permutation

    """

    cdef np.ndarray[ndim=2, dtype=np.float64_t] permuted_tensor
    cdef np.npy_intp i, j, k, n

    n = tensor.shape[0]
    permuted_tensor = np.empty((n, n * n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                permuted_tensor[i, (n * k) + j] = tensor[k, (n * j) + i]
    return permuted_tensor


@cython.boundscheck(False)
@cython.wraparound(False)
def tensor_3d_permute_312(np.ndarray[ndim=2, dtype=np.float64_t, mode="c"] tensor):
    """Permute 3-D tensor with 3-1-2 Mode

    This is a fast version to calculate mode (3,1,2) permutaion of
    a 3-D tensor.

    Parameters
    ----------
    tensor :  array, (n, n * n)
        3D tensor in unfolded format.

    tensor_shape : integer triple
        Shape of the tensor. Since tensor is in
        unfolded format. We need it's real format
        to calculate permutation.

    Returns
    -------
    permuted_tensor: array, (n, n * n)
        Tensor after permutation

    """

    cdef np.ndarray[ndim=2, dtype=np.float64_t] permuted_tensor
    cdef np.npy_intp i, j, k, n

    n = tensor.shape[0]
    permuted_tensor = np.empty((n, n * n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                permuted_tensor[i, (n * k) + j] = tensor[j, (n * i) + k]
    return permuted_tensor
