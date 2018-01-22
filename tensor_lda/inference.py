"""Inference Document Topic Distribution
"""

import numpy as np
import scipy.sparse as sp

from scipy.misc import logsumexp
from sklearn.externals.six.moves import xrange

EPS = np.finfo(np.float).eps


def lda_inference_gd(X, alpha, beta, max_iter, step_size=1e-3, tol=1e-7, smooth=1e-3):
    """LDA Inference with Grandient Descent
    
    This will maximize log(P(theta | X, alpha, beta)).
    
    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Document word matrix.

    alpha : array-like, shape=(n_topics,)
        Prior distribution of topics.

    beta : array-like, shape=(n_topics, n_features)
        Topic word distribution.

    max_iter : int
        Max Iterations of gradient descent
    
    step_size : float, default=1e-3
        Step size of gradient descent method
    
    tol : float, default=1e-7
        Converge condition

    Returns
    -------
    doc_topic_distr : array-like, shape=(n_samples, n_topics)
        Document topic matrix.

    """
    alpha_minus = alpha - 1.
    n_doc = X.shape[0]
    n_components = beta.shape[0]
    is_sparse_x = sp.issparse(X)

    doc_topic_distr = np.ones((n_doc, n_components)) / n_components

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    for idx_d in xrange(n_doc):
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        theta_d = doc_topic_distr[idx_d, :]
        # (k, len(ids))
        beta_d = beta[:, ids]

        theta_d_old = theta_d.copy()
        for _ in xrange(max_iter):
            # shape = (k,)
            grad = alpha_minus / (theta_d + EPS)
            # shape=(len(ids),)
            denom = np.dot(theta_d, beta_d)
            # shape=(k, len(ids))
            part2 = (beta_d / denom) * cnts
            grad += part2.sum(axis=1)

            log_theta_d = np.log(theta_d + EPS) + (step_size * grad)
            logsum_theta_d = logsumexp(log_theta_d)
            theta_d = np.exp(log_theta_d - logsum_theta_d)
            theta_d += smooth
            theta_d /= theta_d.sum()
            if np.abs(theta_d - theta_d_old).mean() < tol:
                break
        doc_topic_distr[idx_d] = theta_d
    return doc_topic_distr


def doc_likelihood(X, theta, alpha, beta):
    """Log Likelihood of document

    This is sum of log(P(theta | X, alpha, beta))
    
    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Document word matrix.
    
    theta : array-like, shape=(n_samples, n_topics)
        Document topic matrix.

    alpha : array-like, shape=(n_topics,)
        Prior distribution of topics.

    beta : array-like, shape=(n_topics, n_features)
        Topic word distribution.
    
    Returns
    -------
    loglikelihood : float
        Sum of Log-likelihood value of P(theta | X, alpha, beta)

    """
    alpha_minus = alpha - 1.
    n_doc = X.shape[0]
    is_sparse_x = sp.issparse(X)

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    # sum of (alpha_t - 1) * log(theta_t) 
    loglikelihood = (np.log(theta) * alpha_minus).sum()
    for idx_d in xrange(n_doc):
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        theta_d = theta[idx_d, :]
        beta_d = beta[:, ids]
        # sum_N {log(sum_T(theta_t * phi_i_t))}
        loglikelihood += (np.log(np.dot(theta_d, beta_d)) * cnts).sum()
    return loglikelihood
