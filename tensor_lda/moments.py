import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp

from sklearn.externals.six.moves import xrange
from sklearn.utils import (check_random_state, check_array,
                           gen_batches, gen_even_slices,
                           _get_n_jobs)
from sklearn.utils.validation import check_non_negative



def first_order_moments(X, min_words, whom):
    """First-Order Moments
    
    First order Moment of document-word frequency matrix.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Matrix of Document-word frequency. `n_samples` is the
        number of documnet and `n_features` are number of unique
        words in the corpus.
    
    min_words : Integer
        Minimum number of words in each document. In LDA, the number
        is 3 since we need 3rd order moments.

    whom : string
        which method called this function.

    Returns
    -------
    e1 : array, shape=(n_features,)
        Expectation of each words in the input matrix.
    
    ignored: integer
        Number of ignored documents.

    """
    X = check_array(X, accept_sparse='csr')
    check_non_negative(X, whom)
    n_samples, n_features = X.shape
    is_sparse_x = sp.issparse(X)


    e1 = np.zeros(n_features)
    doc_word_cnts = np.squeeze(np.asarray(X.sum(axis=1)))
    ignored_docs = 0

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    # TODO: optimize for loop with cython
    for idx_d in xrange(n_samples):
        # get word_id and count in each document
        words_cnt = doc_word_cnts[idx_d]
        if words_cnt < min_words:
            ignored_docs += 1
            continue

        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        for w_id, w_cnt in zip(ids, cnts):
            e1[w_id] += (w_cnt / float(words_cnt))
    e1 /= (n_samples - ignored_docs)

    return (e1, ignored_docs)


def cooccurrence_expectation(X, min_words, whom, batch_size=1000):
    """Expectation of Word Co-occurrence

    Expectation of 2 words occured in the same document.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Matrix of Document-word frequency. `n_samples` is the
        number of documnet and `n_features` are number of unique
        words in the corpus.
    
    min_words : Integer
        Minimum number of words in each document. In LDA, the number
        is 3 since we need 3rd order moments.

    whom : string
        which method called this function.

    Returns
    -------
    e2 : sparse matrix, shape=(n_features, n_features)
        Expectation of word pairs
    
    ignored: integer
        Number of ignored documents.

    """

    X = check_array(X, accept_sparse='csr')
    check_non_negative(X, whom)
    n_samples, n_features = X.shape
    is_sparse_x = sp.issparse(X)

    pairs = []
    doc_word_cnts = np.squeeze(np.asarray(X.sum(axis=1))).reshape(n_samples,)
    ignored_docs = 0

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    # TODO: optimize for loop with cython
    e2 = sp.coo_matrix((n_features, n_features))
    rows = []
    cols = []
    vals = []
    for idx_d in xrange(n_samples):
        # get word_id and count in each document
        words_cnt = doc_word_cnts[idx_d]
        if words_cnt < min_words:
            ignored_docs += 1
            continue

        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]
        unique_word_cnt = len(ids)
        unique_pairs = unique_word_cnt * (unique_word_cnt - 1)
        # index for total tokens are:
        # 0 to (unique_word_cnt - 1): pair(i, i)
        # unique_word_cnt to end: pair(i, j) where (j > i)
        total_non_zeros = unique_word_cnt + unique_pairs
        row_index = np.zeros(total_non_zeros).astype('int')
        col_index = np.zeros(total_non_zeros).astype('int')
        idx_vals = np.zeros(total_non_zeros)

        idx = 0
        for i in xrange(unique_word_cnt):
            cnt = cnts[i]
            # if cnt = 0, val will be 0. don't need to set index
            if cnt > 1:
                row_index[idx] = ids[i]
                col_index[idx] = ids[i]
                idx_vals[idx] = cnt * (cnt - 1)
            idx += 1

        for i in xrange(unique_word_cnt):
            for j in xrange(i + 1, unique_word_cnt):
                row_index[idx] = ids[i]
                col_index[idx] = ids[j]
                idx_vals[idx] = cnts[i] * cnts[j]
                idx += 1
        idx_vals /= (words_cnt * (words_cnt - 1))
        rows.append(row_index)
        cols.append(col_index)
        vals.append(idx_vals)

        # merge for each batch
        if idx_d % batch_size == 0:
            r = np.hstack(rows)
            c = np.hstack(cols)
            data = np.hstack(vals)
            e2 += sp.coo_matrix((data, (r, c)), shape=(n_features, n_features))
            # reset
            rows = []
            cols = []
            vals = []

    # last batch
    if len(rows) > 0:
        r = np.hstack(rows)
        c = np.hstack(cols)
        data = np.hstack(vals)
        e2 += sp.coo_matrix((data, (r, c)), shape=(n_features, n_features))

    # add symmetric pairs to lower triangle
    e2 /= (n_samples - ignored_docs)
    e2 += sp.triu(e2, k=1).T
    return (e2, ignored_docs)


def second_order_moments(n_components, e2, m1, alpha0):
    """Second-Order Moments

    To prevent creating 2nd order moments explicitly, we construct its
    decomposition with `n_components`. check reference [?] section 5.2
    for details.

    Parameters
    ----------
    n_components: int
        Number of components

    e2: sparse matrix, shape=(n_features, n_features)
        Expectation of word pairs. e2[i, j] is the expectation of word `i`
        and `j` in the same document.
    
    m1: array, shape=(n_features,)
        Expectation of each words.

    alpha0: double
        Sum of topic topic concentration parameter
    
    Returns
    -------
    m2_vals : array, shape=(n_components,)
        eigen values of sencond-order moments

    m2_vecs : array, shape=(n_features, n_components)
        eigen values of sencond-order moments        
    """

    # eigen values and vectors of E2
    e2_vals, e2_vecs = sp.linalg.eigsh(e2, k=n_components)
    m1_p = np.dot(e2_vecs.T, m1)
    m2_p = ((-1. * alpha0) / (alpha0 + 1.)) * (m1_p * m1_p[:, np.newaxis])
    m2_p[np.diag_indices_from(m2_p)] += e2_vals

    # eigen values and vectors of M2 prime 
    m2p_vals, m2p_vecs = LA.eigh(m2_p)

    m2_vals = m2p_vals
    m2_vecs = np.dot(e2_vecs, m2p_vecs)

    return (m2_vals, m2_vecs)
