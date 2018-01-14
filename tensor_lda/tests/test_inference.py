import numpy as np

from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_array_almost_equal

from tensor_lda.inference import (lda_inference_gd,
                                  doc_likelihood)
from tensor_lda.utils.sample_generator import LdaSampleGenerator


def test_doc_likelihood():
    rng = np.random.RandomState(2)
    n_topics = rng.randint(15, 20)
    n_words = rng.randint(400, 500)
    mean_words = 10
    min_words = 3
    doc_topic_prior = 1.
    topic_word_prior = 1.
    n_doc = rng.randint(500, 1000)

    gen = LdaSampleGenerator(n_topics=n_topics, n_words=n_words,
                             min_doc_size=min_words,
                             mean_doc_size=mean_words,
                             doc_topic_prior=doc_topic_prior,
                             topic_word_prior=topic_word_prior,
                             random_state=0)

    docs_distr, doc_word_mtx = gen.generate_documents(n_doc)
    alpha = gen.doc_topic_prior_
    beta = gen.topic_word_distr_

    doc_ll_true = doc_likelihood(doc_word_mtx, docs_distr, alpha, beta)

    # uniform distribution
    docs_distr_uniform = np.ones((n_doc, n_topics)) / n_topics
    doc_ll_uniform = doc_likelihood(
        doc_word_mtx, docs_distr_uniform, alpha, beta)
    assert_greater(doc_ll_true, doc_ll_uniform)

def test_lda_inference_gd():
    rng = np.random.RandomState(2)
    n_topics = rng.randint(15, 20)
    n_words = rng.randint(400, 500)
    mean_words = 10
    min_words = 3
    doc_topic_prior = 1.
    topic_word_prior = 1.
    n_doc = rng.randint(500, 1000)

    gen = LdaSampleGenerator(n_topics=n_topics, n_words=n_words,
                             min_doc_size=min_words,
                             mean_doc_size=mean_words,
                             doc_topic_prior=doc_topic_prior,
                             topic_word_prior=topic_word_prior,
                             random_state=0)

    _, doc_word_mtx = gen.generate_documents(n_doc)
    alpha = gen.doc_topic_prior_
    beta = gen.topic_word_distr_

    docs_distr_uniform = np.ones((n_doc, n_topics)) / n_topics
    ll_uniform = doc_likelihood(doc_word_mtx, docs_distr_uniform,
                                alpha, beta)
    inference_1 = lda_inference_gd(doc_word_mtx, alpha, beta,
                                   max_iter=5)
    ones = np.ones(n_doc)
    assert_array_almost_equal(ones, inference_1.sum(axis=1))
    ll_inference_1 = doc_likelihood(doc_word_mtx, inference_1,
                                    alpha, beta)

    inference_2 = lda_inference_gd(doc_word_mtx, alpha, beta, 
                                   max_iter=1000)
    assert_array_almost_equal(ones, inference_2.sum(axis=1))
    ll_inference_2 = doc_likelihood(doc_word_mtx, inference_2,
                                    alpha, beta)
    assert_greater(ll_inference_2, ll_inference_1)
    assert_greater(ll_inference_1, ll_uniform)
