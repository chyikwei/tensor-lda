import numpy as np
import scipy.sparse as sp

from tensor_lda.tensor_lda import TensorLDA

#def test_estimator():
#    """Test estimator"""
#    return check_estimator(TensorLDA)


def test_tensor_lda_simple():
    rng = np.random.RandomState(0)

    n_features = 500
    n_samples = rng.randint(100, 200)
    doc_word_mtx = rng.randint(0, 3, size=(n_samples, n_features))
    doc_word_mtx = sp.csr_matrix(doc_word_mtx)

    lda = TensorLDA(n_components=50, alpha0=10.)
    lda.fit(doc_word_mtx)
