"""Compute CP decomposition for 3-D tensors
"""

import numpy as np

from numpy import linalg as LA
from sklearn.utils import check_random_state
from .utils.tensor_utils import khatri_rao_prod


def _check_3d_tensor(tensor, n_dim):
    
    t_shape = tensor.shape
    n_col = n_dim * n_dim
    if len(t_shape) != 2:
        raise ValueError("dimension mismatch: tensor need be a 2-D array")

    if t_shape[0] != n_dim:
        raise ValueError("row dimension mismatch: %d != %d"
                         % (t_shape[0], n_dim))

    if t_shape[1] != n_col:
        raise ValueError("column dimension mismatch: %d != %d"
                         % (t_shape[1], n_col))


def _als_iteration(tensor, b, c):
    """One ALS iteration"""

    temp1 = np.dot(tensor, khatri_rao_prod(b, c))
    temp2 = LA.pinv(np.dot(c.T, c) * np.dot(b.T, b))
    a_update = np.dot(temp1, temp2)

    lambdas = LA.norm(a_update, axis=0)
    a_update /= lambdas
    return lambdas, a_update


def tensor_reconstruct(lambdas, a, b, c):
    t = np.dot(np.dot(a, np.diag(lambdas)),
               khatri_rao_prod(c, b).T)
    return t


def cp_als(tensor, n_component, n_restart, n_iter, tol, random_state):
    """Cp decomposition for symmertic 3-D tensor with ALS

    CP decomposition with Alternating least square (ALS) method.
    The method assume the tensor can be composed by sum of 
    `n_component` rank one tensors.

    Parameters
    ----------
    tensor : array, (k, k * k)
        Tensor to be decomposed with unfolded
        format.

    n_component: int
        Number of components

    n_restart: int
        Number of ALS restarts

    n_iter: int
        Number of iterations for ALS

    random_state: int, RandomState instance or None, optional, default = None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    tol : float, optional
        Tolerance

    Returns
    -------
    lambda: array, (k,)
    
    a: array, (k, k)

    b: array, (k, k)

    c: array, (k, k)
    """

    # check tensor shape
    _check_3d_tensor(tensor, n_component)

    converge_threshold = 1.0 - tol
    random_state = check_random_state(random_state)

    best_a = None
    best_b = None
    best_c = None
    best_lambdas = None
    best_loss = None

    for i in xrange(n_restart):

        # use QR factorization to get random starts
        a, _ = LA.qr(random_state.randn(n_component, n_component))
        b, _ = LA.qr(random_state.randn(n_component, n_component))
        c, _ = LA.qr(random_state.randn(n_component, n_component))

        for iteration in xrange(n_iter):

            # check convergence
            if iteration > 0:
                diag = np.diag(np.dot(prev_a.T, a))
                #print(diag)
                if np.all(diag > converge_threshold):
                    print("ALS converge in %d iterations" % iteration)
                    break

            prev_a = a
            _, a = _als_iteration(tensor, b, c)
            _, b = _als_iteration(tensor, c, a)
            lambdas, c = _als_iteration(tensor, a, b)

        # update optimal values
        reconstructed = tensor_reconstruct(lambdas, a, b, c)
        loss = LA.norm(tensor - reconstructed)
        print("restart: %d, loss: %.5f" % (i, loss))
        if best_loss is None or loss < best_loss:
            best_a = a
            best_b = b
            best_c = c
            best_lambdas = lambdas
            best_loss = loss
            print("ALS best loss: %.5f" % best_loss)
            if best_loss < tol:
                break

    return (best_lambdas, best_a, best_b, best_c)
