"""Utility functions for tensor operations
"""

import numpy as np


def _check_1d_vector(vector):
    """Check 1D vector shape

    Check 1D vector shape. array with shape
    [n, 1] or [n, ] are accepted. Will return
    a 1 dimension vector.

    Parameters
    ----------
    vector : array [n,] or [n, 1]
        rank one vector
    
    Returns
    -------
    vector : array, [n,
    """

    v_shape = vector.shape
    if len(v_shape) == 1:
        return vector
    elif len(v_shape) == 2 and v_shape[1] == 1:
        return vector.reshape(v_shape[0],)
    else:
        raise ValueError("Vector is not 1-d array: shape %s" % str(v_shape))


def rank_1_tensor_3d(a, b, c):
    """Create rank one 3-D tensor from vectors

    Create unfolded 3D tensor from 3 rank one
    vectors `a`, `b`, and `c`. The return value
    is an unfolded 3D tensor.

    Parameters
    ----------
    a : array, (n,)
        rank one vector

    b :  array, (n,)
        rank one vector

    c :  array, (n,)
        rank one vector
    
    Returns
    -------
    tensor:  array, (n, n * n)
        3D tensor in unfolded format. element
        (i, j, k) will map to (i, (n * k) + j)
    """

    a = _check_1d_vector(a)
    b = _check_1d_vector(b)
    c = _check_1d_vector(c)

    dim = a.shape[0]
    # check dimension
    if (dim != b.shape[0]) or (dim != c.shape[0]):
        raise ValueError("Vector dimension mismatch: (%d, %d, %d)" %
                         (dim, b.shape[0], c.shape[0]))

    outter = b[:, np.newaxis] * c[:, np.newaxis].T
    tensor = a[:, np.newaxis] * outter.ravel(order='F')[:, np.newaxis].T
    return tensor


def khatri_rao_prod(a, b):
    """Calculate Khatri-Rao product
    calculate khatri_rao_prod of matix 'a' and 'b'
    

    """
