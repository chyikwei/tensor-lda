============
Introduction
============

`TensorLDA` uses tensor decomposition method to estimate parameters in a Latent Dirichlet Allocation model.
For an intruduction about how LDA works, check scikit-learn's `LatentDirichletAllocation` document. This document will focus on how to learn LDA parameters by using tensor decomposition method.

LDA Parameters
--------------

First, we define LDA parameters:

* :math:`K` : number of topics

* :math:`V` : vocabulary size

* :math:`\alpha_i (i = 1...K)` : dirichlet prior for topic i

* :math:`\alpha_0` : sum of topic priors (:math:`\alpha_0 = \sum_{i=1}^{K} \alpha_i`)

* :math:`\beta_i (i = 1...K)` : word distribution for topic i

In the model we assume :math:`\alpha_0` is given and our goal is to estimate :math:`\alpha_i` and :math:`beta_i` from a given corpus.

Data Moments
------------

Let :math:`x_1`, :math:`x_2`, and :math:`x_3` be any triple of words in a document, the first three order data moments :math:`M_1`, :math:`M_2`, and :math:`M_3` are defined as:

.. math::

  M_1 = & \mathop{\mathbb{E}}[x_1] \\ 

  M_2 = & \mathop{\mathbb{E}}[x_1 \otimes x_2] - \frac{\alpha_0}{\alpha_0 + 1} M_1 \otimes M_1 \\


  M_3 = & \mathop{\mathbb{E}}[x_1 \otimes  x_2 \otimes x_3] \\
         & - \frac{\alpha_0}{\alpha_0 + 2} (
              \mathop{\mathbb{E}}[x_1 \otimes x_2 \otimes M_1] +
              \mathop{\mathbb{E}}[x_1 \otimes M_1 \otimes x_3] + 
              \mathop{\mathbb{E}}[M_1 \otimes x_2 \otimes x_3]) \\
        & + \frac{\alpha_0^2}{(\alpha_0 + 2)(\alpha_0 + 1)} M_1 \otimes M_1 \otimes M_1


Based on dirichelet priors, we can derive the relationship between data moments and model parameters as:

.. math::
  
  M_2 = & \sum_{i=1}^{k} \alpha_i \beta_i \otimes \beta_i \\

  M_3 = & \sum_{i=1}^{k} \alpha_i \beta_i \otimes \beta_i \otimes \beta_i


Whitening
---------

TODO

.. math::
  
  T = \mathop{\mathbb{E}}[W^\top x_1 \otimes  W^\top x_2 \otimes W^\top x_3]


Tensor Decomposition
--------------------

TODO



.. topic:: References:

    * `"Tensor Decompositions for Learning Latent Variable Models"
      <http://www.cs.columbia.edu/~djhsu/papers/power-jmlr.pdf>`_
      A. Anandkumar, R. Ge, D. Hsu, S. M. Kakade, M. Telgarsky, 2014

    * `"Scalable Moment-Based Inference for Latent Dirichlet Allocation"
      <https://www.microsoft.com/en-us/research/wp-content/uploads/2014/09/ecmlpkdd14STOD.pdf>`_
      C. Wang, X. Liu, Y. Song, and J. Han, 2014

    * `"Latent Dirichlet Allocation"
      <http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf>`_
      D. Blei, A. Ng, M. Jordan, 2003

    * `"Latent Dirichlet Allocation (Scikit-learn document)"
      <http://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda>`_
