#!/usr/bin/env python
from distutils.core import setup

__version__ = '0.1'

setup(name = 'nonGaussLike',
      version = __version__,
      description = 'non-Gaussian Likelihood',
      author='ChangHoon Hahn',
      author_email='hahn.changhoon@gmail.com',
      url='',
      platforms=['*nix'],
      license='GPL',
      requires = ['numpy', 'matplotlib', 'scipy'],
      provides = ['nongausslike'],
      packages = ['nongausslike']
      )
