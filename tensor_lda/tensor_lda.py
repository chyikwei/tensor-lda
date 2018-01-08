"""Latent Dirichlet Allocation with tesnsor Decomposition

TODO: add short desc
"""

# Author: Chyi-Kwei Yau

from __future__ import print_function

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_random_state, check_array

from sklearn.utils.validation import check_non_negative

from .moments import (first_order_moments,
                      cooccurrence_expectation,
                      second_order_moments,
                      third_order_monents,
                      whitening,
                      unwhitening)

from .cp_decompose import cp_als


class TensorLDA(BaseEstimator, TransformerMixin):
    """Latent Dirichlet Allocation with tensor decomposition

    Parameters
    ----------
    n_components : int, optional (default=10)
        Number of topics.

    alpha0 : double, optional (default=10.)
        Sum of topic prior alpha.

    max_iter : integer, optional (default=10)
        The maximum number of iterations.
    
    converge_tol : float, optional (default=1e-4)
        Convergence tolarence.

    verbose : int, optional (default=0)
        Verbosity level.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    
    Attributes
    ----------
    components_ : array, [n_components, n_features]
        parameters for topic word distribution.
    
    alpha_ : array, [n_components]

    
    References
    ----------
    [1] "Tensor Decompositions for Learning Latent Variable Models",
        Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M. Kakade,
        Matus Telgarsky, 2014

    [2] "Scalable Moment-Based Inference for Latent Dirichlet Allocation",
        Chi Wang, Xueqing Liu, Yanglei Song, and Jiawei Han, 2014

    [3] "Tensor Decompositions and Applications", Tamara G. Kolda,
        Brett W. Bader, 2009
    """

    def __init__(self, n_components=10, alpha0=10., max_iter=100,
                 n_restart=10, converge_tol=1e-4, verbose=0,
                 random_state=None):
        self.n_components = n_components
        self.alpha0 = alpha0
        self.max_iter = max_iter
        self.n_restart = n_restart
        self.converge_tol = converge_tol
        self.verbose = verbose
        self.random_state = random_state

        # latent variables
        self.components_ = None
        self.alpha_ = None
        self.random_state_ = check_random_state(self.random_state)

    def _check_params(self):
        """Check model parameters."""
        if self.n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r"
                             % self.n_components)

        if self.alpha0 <= 0:
            raise ValueError("Invalid 'alpha0' parameter: %r"
                             % self.alpha0)

    def _check_non_neg_array(self, X, whom):
        """check X format
        check X format and make sure no negative value in X.
        Parameters
        ----------
        X :  array-like or sparse matrix
        """
        X = check_array(X, accept_sparse='csr')
        check_non_negative(X, whom)
        return X

    def fit(self, X, y=None):
        """Learn model for the data X with tensor decomposition method

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.

        y : Ignored

        Returns
        -------
        self       
        """
        self._check_params()
        X = self._check_non_neg_array(X, "TensorLDA.fit")

        n_components = self.n_components
        alpha0 = self.alpha0
 
        # 1st order moments
        m1, ignored_doc_cnt = first_order_moments(X, 3)

        # 2nd order moments
        e2, _ = cooccurrence_expectation(X, 3, batch_size=1000)
        m2_vals, m2_vecs = second_order_moments(
            n_components, e2, m1, alpha0)

        # whitening matrix
        w_matrix = whitening(m2_vals, m2_vecs)
        uw_matrix = unwhitening(m2_vals, m2_vecs)

        # 3rd order moments
        m3 = third_order_monents(X, w_matrix, m1, alpha0)

        # ALS decomposition
        e_vals, e_vecs, e_vecs_2, e_vecs_3 = cp_als(m3, n_components,
                                      n_restart=self.n_restart,
                                      n_iter=self.max_iter,
                                      tol=self.converge_tol,
                                      random_state=self.random_state_)
        # unwhitening
        self.cp_results_ = {
            'e_vals': e_vals,
            'e_vecs': e_vecs,
            'e_vecs_2': e_vecs_2,
            'e_vecs_3': e_vecs_3
        }
        self.components_ = (np.dot(uw_matrix, e_vecs) * e_vals).T
        self.alpha_ = alpha0 * np.power(e_vals, -2)

    def transform(self, X):
        pass
