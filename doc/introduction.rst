==================================================================
Introduction
==================================================================

`TensorLDA` implements a tensor based method to estimate parameters in a Latent Dirichlet Allocation(LDA) model. This document will focus on the parameter estimation. For more detais about Latent Dirichlet Allocation model, check scikit-learn's `LatentDirichletAllocation` document.

Assuming :math:`x_1`, :math:`x_2`, and :math:`x_3` are any triple of words in the same document,
the first three order data moments :math:`M_1`, :math:`M_2`, and :math:`M_3` are defined as:

.. math::

  M_1 = & \mathop{\mathbb{E}}[x_1] \\ 


  M_2 = & \mathop{\mathbb{E}}[x_1 \otimes x_2] - \frac{\alpha_0}{\alpha_0 + 1} M_1 \otimes M_1 \\


  M_3 = & \mathop{\mathbb{E}}[x_1 \otimes  x_2 \otimes x_3] \\
         & - \frac{\alpha_0}{\alpha_0 + 2} (
              \mathop{\mathbb{E}}[x_1 \otimes x_2 \otimes M_1] +
              \mathop{\mathbb{E}}[x_1 \otimes M_1 \otimes x_3] + 
              \mathop{\mathbb{E}}[M_1 \otimes x_2 \otimes x_3]) \\
        & + \frac{\alpha_0^2}{(\alpha_0 + 2)(\alpha_0 + 1)} M_1 \otimes M_1 \otimes M_1


Model Parameter:

.. math::
  
  M_2 = & \sum_{i=1}^{k} \alpha_i \beta_i \otimes \beta_i \\

  M_3 = & \sum_{i=1}^{k} \alpha_i \beta_i \otimes \beta_i \otimes \beta_i


Whitening:

.. math::
  
  T = \mathop{\mathbb{E}}[W^\top x_1 \otimes  W^\top x_2 \otimes W^\top x_3]


Tensor Decomposition:



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
