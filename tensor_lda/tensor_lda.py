"""Latent Dirichlet Allocation with Tesnsor Decomposition
"""

# Author: Chyi-Kwei Yau

from __future__ import print_function

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_non_negative

from .moments import (first_order_moments,
                      cooccurrence_expectation,
                      second_order_moments,
                      third_order_monents,
                      whitening,
                      unwhitening)

from .cp_decompose import cp_tensor_power_method
from .inference import lda_inference_vi


class TensorLDA(BaseEstimator, TransformerMixin):
    """Latent Dirichlet Allocation with tensor decomposition

    Parameters
    ----------
    n_components : int, optional (default=10)
        Number of topics.

    alpha0 : double, optional (default=0.1)
        Sum of topic prior alpha.

    max_iter : integer, optional (default=100)
        The maximum number of iterations.

    max_inference_iter : integer, optional (default=1000)
        The maximum number of inference iterations.
    
    converge_tol : float, optional (default=1e-4)
        Convergence tolarence in training step.

    verbose : int, optional (default=0)
        Verbosity level.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Attributes
    ----------
    n_components_ : int
        The effective number of components. The number must be smaller or equal
        to the number of features.

    components_ : array, [n_components, n_features]
        Tarameters for topic word distribution.
    
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

    def __init__(self, n_components=10, alpha0=0.1, max_iter=1000,
                 max_inference_iter=1000, n_restart=10,
                 converge_tol=1e-4, inference_converge_tol=1e-6,
                 inference_step_size=1e-3, verbose=0,
                 smooth_param=0.01,
                 random_state=None):
        self.n_components = n_components
        self.alpha0 = alpha0
        self.max_iter = max_iter
        self.max_inference_iter = max_inference_iter
        self.n_restart = n_restart
        self.converge_tol = converge_tol
        self.inference_converge_tol = inference_converge_tol
        self.inference_step_size = inference_step_size
        self.verbose = verbose
        self.smooth_param = smooth_param
        self.random_state = random_state

    def _check_params(self, X):
        """Check model parameters."""
        if self.n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r"
                             % self.n_components)

        if self.alpha0 <= 0:
            raise ValueError("Invalid 'alpha0' parameter: %r"
                             % self.alpha0)

        n_features = X.shape[1]
        n_components = self.n_components
        if n_features < self.n_components:
            # add warning
            self.n_components_ = n_features
        else:
            self.n_components_ = n_components

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

    def _check_inference(self, X, whom):
        """Check inference conditions

        The function will check model is fitted and input matrix
        has correct shape & value
        """
        if not hasattr(self, "components_"):
            raise NotFittedError("no 'components_' attribute in model."
                                 " Please fit model first.")

        X = self._check_non_neg_array(X, whom)

        n_features = X.shape[1]
        if n_features != self.components_.shape[1]:
            raise ValueError(
                "The provided data has %d dimensions while "
                "the model was trained with feature size %d." %
                (n_features, self.components_.shape[1]))
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
        random_state = check_random_state(self.random_state)
        X = self._check_non_neg_array(X, "TensorLDA.fit")
        self._check_params(X)

        n_components = self.n_components_
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
        e_vals, e_vecs, n_iter = cp_tensor_power_method(m3, n_components,
                                                        self.n_restart,
                                                        self.max_iter,
                                                        random_state)
        self.n_iter_ = n_iter
        #self.cp_results_ = {
        #    'e_vals': e_vals,
        #    'e_vecs': e_vecs,
        #}
        # mormalize alpha to sum to alpha0
        alpha = np.power(e_vals, -2)
        self.alpha_ = (alpha / alpha.sum()) * alpha0

        # unwhitening
        # TODO: use L1 max
        components = (np.dot(uw_matrix, e_vecs) * e_vals).T
        # set negative part to 0
        components[components < 0.] = 0.
        # smooth beta
        components *= (1. - self.smooth_param)
        components += (self.smooth_param / components.shape[1])

        components /= components.sum(axis=1)[:, np.newaxis]
        self.components_ = components
        return self

    def transform(self, X):
        """Transform data X according to the fitted model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.


        Returns
        -------
        doc_topic_distr : shape=(n_samples, n_topics)
            document topic distribution for X.

        """
        X = self._check_inference(X, "TensorLDA.transform")
        alpha = self.alpha_
        beta = self.components_
        max_iters = self.max_inference_iter
        tol = self.inference_converge_tol

        doc_topic_distr = lda_inference_vi(X, alpha, beta, max_iters, tol)
        doc_topic_distr /= doc_topic_distr.sum(axis=1)[:, np.newaxis]
        #doc_topic_distr = lda_inference_gd(X, alpha, beta,
        #                                   max_iter, step_size, tol)
        return doc_topic_distr
