import numpy as np
import scipy.sparse as sp

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises_regexp
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_true

from sklearn.externals.six.moves import xrange

from tensor_lda.moments import (first_order_moments,
                                cooccurrence_expectation,
                                second_order_moments,
                                whitening,
                                unwhitening,
                                whitening_triples_expectation)

from tensor_lda.utils.tensor_utils import tensor_3d_prod


def _triples_expectation(X):
    # calculate the exact triple words expectation
    # this will generate a (n_features, n_features * n_features)
    # matrix

    n_samples, n_features = X.shape
    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    ignored_cnt = 0
    e_triples = np.zeros((n_features, n_features, n_features))

    for idx_d in xrange(n_samples):
        # get word_id and count in each document
        ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]

        unique_ids = len(ids)
        total = cnts.sum()
        coef = 1. / (total * (total - 1.) * (total - 2.))
        # min word count for triples in a doc is 3.
        # ignore others
        if total < 3:
            ignored_cnt += 1
            continue

        for i in xrange(unique_ids):
            cnt_i = cnts[i]
            for j in xrange(unique_ids):
                cnt_j = cnts[j]
                for k in xrange(unique_ids):
                    cnt_k = cnts[k]
                    # case_1: i = j = k
                    if i == j and j == k:
                        if cnt_i >= 3:
                            combinations = cnt_i * (cnt_i - 1.) * (cnt_i - 2.)
                        else:
                            combinations = 0.
                    # case_2: i = j, j != k
                    elif i == j and j != k:
                        combinations = cnt_i * (cnt_i - 1.) * cnt_k
                    # case_3: j = k, i != j
                    elif j == k and i != j:
                        combinations = cnt_j * (cnt_j - 1.) * cnt_i
                    # case_4: i = k, j != k
                    elif i == k and j != k:
                        combinations = cnt_i * (cnt_i - 1.) * cnt_j
                    # case_5: i != k, j != k, i != k
                    else:
                        combinations = cnt_i * cnt_j * cnt_k
                    e_triples[i, j, k] += (coef * combinations)
    e_triples /= (n_samples - ignored_cnt)
    return e_triples


def test_first_order_moments():
    rng = np.random.RandomState(0)

    n_features = rng.randint(100, 500)
    n_samples = rng.randint(1000, 2000)
    doc_word_mtx = rng.randint(0, 10, size=(n_samples, n_features))

    word_cnts = doc_word_mtx.sum(axis=1).astype('float')
    result = (doc_word_mtx / word_cnts[:, np.newaxis]).sum(axis=0)
    result /= n_samples
    m1, ignored_cnt = first_order_moments(doc_word_mtx, min_words=0,
                             whom='test_first_order_moments')
    assert_equal(0, ignored_cnt)
    assert_array_almost_equal(result, m1)


def test_first_order_moments_with_ignored_count():
    rng = np.random.RandomState(0)

    n_features = 10
    n_samples = rng.randint(2000, 4000)
    doc_word_mtx = rng.randint(0, 3, size=(n_samples, n_features))

    word_cnts = doc_word_mtx.sum(axis=1).astype('float')
    min_count = int(word_cnts.min() + 1)
    mask = (word_cnts >= min_count)

    result = (doc_word_mtx / word_cnts[:, np.newaxis])[mask, :].sum(axis=0)
    result /= mask.sum()
    m1, ignored_cnt = first_order_moments(doc_word_mtx, min_words=min_count,
                             whom='test_first_order_moments')
    assert_greater(ignored_cnt, 0)
    assert_equal(mask.sum(), n_samples - ignored_cnt)
    assert_array_almost_equal(result, m1)

    # sparse matrix should return same result
    m1_2, ignored_cnt_2 = first_order_moments(sp.csr_matrix(doc_word_mtx),
                        min_words=min_count, whom='test_first_order_moments')
    assert_equal(ignored_cnt, ignored_cnt_2)
    assert_array_almost_equal(m1, m1_2)


def test_cooccurrence_expectation_simple():

    doc_word_mtx = np.array([
        [1, 0, 3, 2],
        [4, 1, 5, 0]
    ])

    result_1 = np.array([
        [0, 0, 3, 2],
        [0, 0, 0, 0],
        [3, 0, 6, 6],
        [2, 0, 6, 2],
    ])

    result_2 = np.array([
        [12, 4, 20, 0],
        [4, 0, 5, 0],
        [20, 5, 20, 0],
        [0, 0, 0, 0],
    ])

    result = (result_1 / float(6 * 5)) + \
             (result_2 / float(10 * 9))
    result /= 2
    e2, ignored_cnt = cooccurrence_expectation(doc_word_mtx, min_words=3,
                             whom='test_cooccurrence_expectation')

    assert_equal(ignored_cnt, 0)
    assert_array_almost_equal(result, e2.toarray())


def test_cooccurrence_expectation():
    rng = np.random.RandomState(0)

    n_features = 100
    n_samples = rng.randint(100, 200)
    doc_word_mtx = rng.randint(0, 3, size=(n_samples, n_features)).astype('float')

    word_cnts = doc_word_mtx.sum(axis=1).astype('float')
    min_count = int(word_cnts.min() + 1)
    mask = (word_cnts >= min_count)

    result = np.zeros((n_features, n_features))
    for i in xrange(n_samples):
        cnt = word_cnts[i]
        if cnt < min_count:
            continue
        doc_i = doc_word_mtx[i, :]
        result_i = (doc_i * doc_i[:, np.newaxis]) - np.diag(doc_i)
        result_i /= cnt * (cnt - 1)
        result += result_i
    result /= mask.sum()

    e2, ignored_cnt = cooccurrence_expectation(doc_word_mtx, min_words=min_count,
                             whom='test_cooccurrence_expectation')

    assert_greater(ignored_cnt, 0)
    assert_equal(mask.sum(), n_samples - ignored_cnt)
    assert_array_almost_equal(result, e2.toarray())
    # cooccurrence should be symmertic
    assert_array_almost_equal(result, e2.toarray().T)


def test_second_order_moments():
    # compare create M2 directly vs create eigen value
    # and vectors with optimized method
    rng = np.random.RandomState(100)

    n_features = 500
    n_components = 50
    min_count = 3
    alpha0 = 10.
    n_samples = rng.randint(100, 150)
    doc_word_mtx = rng.randint(0, 3, size=(n_samples, n_features)).astype('float')
    doc_word_mtx = sp.csr_matrix(doc_word_mtx)

    m1, _ = first_order_moments(doc_word_mtx, min_words=min_count,
                                whom='test_second_order_moments')
    e2, _ = cooccurrence_expectation(doc_word_mtx, min_words=min_count,
                                     whom='test_second_order_moments')

    # create M2 directly
    m2 = (alpha0 + 1.) * e2.toarray()
    m2 -= (alpha0 * m1) * m1[:, np.newaxis]
    m2_vals_true, m2_vecs_true = sp.linalg.eigsh(m2, k=n_components)

    # create M2 eigen values & vectors with optimized method
    m2_vals, m2_vecs = second_order_moments(n_components, e2, m1, alpha0)

    # make sure all eigen values are greater than 0.
    assert_true(np.all(m2_vals > 0.))
    assert_equal(m2_vals.shape[0], n_components)
    assert_equal(m2_vecs.shape[0], n_features)
    assert_equal(m2_vecs.shape[1], n_components)

    m2_reconstruct_true = np.dot(np.dot(m2_vecs_true, np.diag(m2_vals_true)), m2_vecs_true.T)
    m2_reconstruct = np.dot(np.dot(m2_vecs, np.diag(m2_vals)), m2_vecs.T)

    # compare reconstructed version
    assert_array_almost_equal(m2_reconstruct_true, m2_reconstruct)

    # compare original M2 with reconstructed version
    assert_array_almost_equal(m2, m2_reconstruct, decimal=4)


def test_whitening():

    rng = np.random.RandomState(1)

    n_features = 500
    n_components = 50
    min_count = 3
    alpha0 = 10.
    n_samples = rng.randint(100, 150)
    doc_word_mtx = rng.randint(0, 3, size=(n_samples, n_features)).astype('float')
    doc_word_mtx = sp.csr_matrix(doc_word_mtx)

    m1, _ = first_order_moments(doc_word_mtx, min_words=min_count,
                                whom='test_whitening')
    e2, _ = cooccurrence_expectation(doc_word_mtx, min_words=min_count,
                                     whom='test_whitening')

    # create M2 directly
    m2 = (alpha0 + 1.) * e2.toarray()
    m2 -= (alpha0 * m1) * m1[:, np.newaxis]
    m2_vals, m2_vecs = sp.linalg.eigsh(m2, k=n_components)
    # create whitening matrix
    W = whitening(m2_vals, m2_vecs)

    # check whitening matrix shape
    assert_equal(n_features, W.shape[0])
    assert_equal(n_components, W.shape[1])

    # M2(W, W) should be identity matrix
    identity = np.dot(np.dot(W.T, m2), W)
    assert_array_almost_equal(np.eye(n_components, n_components), identity)


def test_whitening_unwhitening():
    rng = np.random.RandomState(12)

    n_features = 500
    n_components = 50
    min_count = 3
    alpha0 = 10.
    n_samples = rng.randint(100, 150)
    doc_word_mtx = rng.randint(0, 3, size=(n_samples, n_features)).astype('float')
    doc_word_mtx = sp.csr_matrix(doc_word_mtx)

    m1, _ = first_order_moments(doc_word_mtx, min_words=min_count,
                                whom='test_whitening_unwhitening')
    e2, _ = cooccurrence_expectation(doc_word_mtx, min_words=min_count,
                                     whom='test_whitening_unwhitening')

    # create M2 directly
    m2 = (alpha0 + 1.) * e2.toarray()
    m2 -= (alpha0 * m1) * m1[:, np.newaxis]
    m2_vals, m2_vecs = sp.linalg.eigsh(m2, k=n_components)
    # create whitening matrix
    W = whitening(m2_vals, m2_vecs)
    uW = unwhitening(m2_vals, m2_vecs)
    identity = np.dot(W.T, uW)
    assert_array_almost_equal(np.eye(n_components, n_components), identity)


def _test_whitening_triples_expectation():
    # WIP

    rng = np.random.RandomState(3)

    n_features = 100
    n_components = 20
    n_samples = 50
    doc_word_mtx = rng.randint(0, 3, size=(n_samples, n_features)).astype('float')
    doc_word_mtx = sp.csr_matrix(doc_word_mtx)

    # random matrix used as whitening matrix
    W = rng.rand(n_features, n_components)

    # compute E3(W, W, W) with optimized method
    e3_w = whitening_triples_expectation(doc_word_mtx, 3, W)

    # create E3
    e3 = _triples_expectation(doc_word_mtx)
    # compute E3(W, W, W) directly
    e3_w_true = tensor_3d_prod(e3, W, W, W)
    # flatten
    e3_w_true_flatten = np.hstack([e3_w_true[:, :, i] for i in xrange(n_components)])

    assert_array_almost_equal(e3_w_true_flatten, e3_w)
