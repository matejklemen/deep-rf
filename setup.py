from setuptools import setup
from warnings import warn
import sys

if sys.version_info.major < 3:
    warn("This implementation was made with Python 3 in mind, so it likely won't work with Python 2")

setup(
   name='gcforest',
   version='0.1a',
   description='Implementation of gcForest',
   author='Matej Klemen',
   packages=['gcforest'],
   install_requires=['numpy', 'sklearn']
)
