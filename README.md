[![Build Status](https://travis-ci.org/chyikwei/tensor-lda.svg?branch=master)](https://travis-ci.org/chyikwei/tensor-lda)
[![Build Status](https://circleci.com/gh/chyikwei/tensor-lda.png?&style=shield)](https://circleci.com/gh/gh/chyikwei/tensor-lda)
[![Coverage Status](https://coveralls.io/repos/github/chyikwei/tensor-lda/badge.svg?branch=master)](https://coveralls.io/github/chyikwei/tensor-lda?branch=master)

Tensor LDA
==========

tensor-lda is a LDA (Latent Dirichlet Allocation) implementation that use tensor decomposition method to estimate LDA parameters. It follows scikit-learn's API and can be used as scikit-learn's extension.

HTML Documentation - https://chyikwei.github.io/tensor-lda/


Install:
--------

```
# clone repoisitory
git clone git@github.com:chyikwei/tensor-lda.git
cd tensor-lda

# install numpy & scipy
pip install -r requirements.txt
pip install .
```

Example:
--------

The usage is the same as scikit-learn's LDA model.

[Here](https://chyikwei.github.io/tensor-lda/auto_examples/tensor_lda_with_20_news_group.html#) is an example that extract topics from 20 news grooup dataset.



Running Test:
-------------
```
python setup.py test
```
