import numpy as np

#from sklearn.utils.testing import assert_array_equal
#from sklearn.utils.testing import assert_true
#from sklearn.utils.testing import assert_false
from sklearn.externals.six.moves import xrange
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises_regexp

from tensor_lda.utils.tensor_utils import (_check_1d_vector,
                                           rank_1_tensor_3d)


def test_check_1d_vectors():
    # test check_1d_vectors function
    rng = np.random.RandomState(0)

    dim = rng.randint(100)
    # check (dim, )
    a = _check_1d_vector(np.ones((dim,)))
    assert_equal(1, len(a.shape))
    assert_equal(dim, a.shape[0])

    # check (dim, 1)
    b = _check_1d_vector(np.ones((dim, 1)))
    assert_equal(1, len(b.shape))
    assert_equal(dim, b.shape[0])

    # check (dim, 2)
    c = np.ones((dim, 2))
    assert_raises_regexp(ValueError, r"^Vector is not 1-d array:",
                         _check_1d_vector, c)


def test_create_3d_rank_1_tensor_simple():
    # test create_3d_rank_1_tensor

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9])
    result = np.array(
        [[28, 35, 42, 32, 40, 48, 36, 45, 54],
         [56, 70, 84, 64, 80, 96, 72, 90, 108],
         [84, 105, 126, 96, 120, 144, 108, 135, 162]])

    tensor = rank_1_tensor_3d(a, b, c)
    assert_array_equal(result, tensor)


def test_create_3d_rank_1_tensor_random():
    # test create_3d_rank_1_tensor with random values
    rng = np.random.RandomState(0)

    dim = rng.randint(20) + 20
    a = rng.rand(dim)
    b = rng.rand(dim)
    c = rng.rand(dim)

    tensor = rank_1_tensor_3d(a, b, c)
    for i in xrange(dim):
        for j in xrange(dim):
            for k in xrange(dim):
                true_val = a[i] * b[j] * c[k]
                tensor_val = tensor[i, (dim * k) + j]
                assert_almost_equal(true_val, tensor_val)
