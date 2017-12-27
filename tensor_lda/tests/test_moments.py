import numpy as np
import scipy.sparse as sp

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises_regexp
from sklearn.utils.testing import assert_greater

from tensor_lda.moments import first_order_moments


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
    result /= n_samples
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
