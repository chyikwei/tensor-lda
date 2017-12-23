import numpy as np

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises_regexp

from sklearn.externals.six.moves import xrange

from scipy.stats import ortho_group
from tensor_lda.cp_decompose import cp_als, tensor_reconstruct
from tensor_lda.utils.tensor_utils import rank_1_tensor_3d


def test_cp_als():
    # test decompose 3D symmertic tensor from random orthogonal vectors

    rng = np.random.RandomState(0)
    dim = rng.randint(10, 15)
    # create orthogonal vectors
    vectors = ortho_group.rvs(dim, random_state=rng)

    # tensor = sum of rank one tensors
    tensor = np.zeros((dim, dim * dim))
    for i in xrange(dim):
        vector = vectors[:, i]
        tensor += rank_1_tensor_3d(vector, vector, vector)

    # running als decompose
    lambdas, a, b, c = cp_als(tensor, dim, 10, 100, 0.001, rng)

    assert_equal(dim, len(lambdas))
    assert_equal((dim, dim), a.shape)
    assert_equal((dim, dim), b.shape)
    assert_equal((dim, dim), c.shape)
    #print("lambda %r" % lambdas)
    #print("a %r" % a)
    #print("b %r" % b)
    #print("c %r" % c)

    for matrix in [a, b, c]:
        for col1 in xrange(dim):
            v = matrix[:, col1]
            assert_almost_equal(1., np.dot(v, v))
            for col2 in range(col1 + 1, dim):
                #print("lambda2 %.3f" % lambdas[col2])
                #print("%d, %d" % (col1, col2))
                assert_almost_equal(0., np.dot(v, matrix[:, col2]))
    reconstruct_tensor = tensor_reconstruct(lambdas, a, b, c)
    assert_array_almost_equal(tensor, reconstruct_tensor, decimal=6)
