import numpy as np
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
    e1 /= n_samples

    return (e1, ignored_docs)
