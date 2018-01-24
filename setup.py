from __future__ import print_function

import sys
import os
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

libraries = []
if os.name == 'posix':
    libraries.append('m')

extensions = [
    Extension("tensor_lda.utils.fast_tensor_ops",
              ["tensor_lda/utils/fast_tensor_ops.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries),
    Extension("tensor_lda._inference",
              ["tensor_lda/_inference.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=libraries)
]

setup(
    name='tensor_lda',
    version='0.0.1',
    description='tensor decomposition for LDA',
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    author='Chyi-Kwei Yau',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    author_email='chyikwei.yau@gmail.com',
)
