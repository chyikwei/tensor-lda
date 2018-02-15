"""
================================
Topic extraction with Tensor LDA
================================

This example is modified from scikit-learn's "Topic extraction with
Non-negative Matrix Factorization and Latent Dirichlet Allocation"
example.

This example applies :class:`tensor_lda.tensor_lda.TensorLDA`
on the 20 news group dataset and the output is a list of topics, each
represented as a list of terms (weights are not shown).


"""

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

from tensor_lda.tensor_lda import TensorLDA

n_samples = 10000
n_features = 1000
n_components = 40
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        topic_prior = model.alpha_[topic_idx]
        message = "Topic #%d (prior: %.3f): " % (topic_idx, topic_prior)
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=2,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:n_samples]
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.8, min_df=5,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()

print("Fitting TensorLDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))

lda = TensorLDA(n_components=n_components, alpha0=.1)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

doc_topics = lda.transform(tf[0:2, :])
print(doc_topics[0, :])
print(data_samples[0])
