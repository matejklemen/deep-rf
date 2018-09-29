# deep-rf

Implementation of deep forest \[1\] in Python 3.  
Also contains a few new features, which are turned off by default:
- random subspace forests \[2\],
- X-of-N trees and random X-of-N forests \[3\],
- new type of "aggregation" (ending) layer (which is used to make predictions
from last layer's features).

Original paper on gcForest is by Zhi-Hua Zhou and Ji Feng and can be found at
[https://arxiv.org/abs/1702.08835](https://arxiv.org/abs/1702.08835).

## Setup
First, clone the repository.  
```
$ git clone https://github.com/matejklemen/deep-rf
$ cd deep-rf
```  
Install the required packages.
```
$ pip install -r requirements.txt 
```
*OPTIONAL: Install the optional packages, which are only used to load the MNIST data set (will be
needed to run the `examples/mnist_base.py` example)*
```
$ pip install -r requirements-opt.txt
```

Then run *setup.py*.  
```
$ python setup.py install
```

## Tests

Run the tests from root of project:
```
$ python -m unittest
```

## Examples
The following example builds a deep forest that consists only of cascade forest (no
multi-grained scanning). Each layer contains 4 random, 4 completely random and 4 random
subspace forests. Each of these forests contains 500 trees. Anywhere k-fold
cross-validation is used in the deep forest algorithm, k = 3. The predictions of models
in last layer are combined using stacking. The example uses LETTER data set for fitting
and predicting.

```python
import numpy as np
from gcforest.gc_forest import GrainedCascadeForest
from gcforest import datasets

train_X, train_y, test_X, test_y = datasets.prep_letter()
gcf = GrainedCascadeForest(n_rf_cascade=4,
                           n_crf_cascade=4,
                           n_rsf_cascade=4,
                           n_estimators_rf=500,
                           n_estimators_crf=500,
                           n_estimators_rsf=500,
                           end_layer_cascade="stack",
                           k_cv=3)

# fit_predict(...) does 'simultaneous' fitting and predicting, i.e. trains a part of forest
# and right after that uses the part to predict intermediate probabilities for the built part
# this is done so that the built models do not need to be kept in memory (as they can quickly
# grow beyond your computer's memory capacity)
preds = gcf.fit_predict(train_X, train_y, test_feats=test_X)
ca = np.sum(preds == test_y) / test_y.shape[0]

print("Classification accuracy: %.5f" % ca)
```

The `examples/` folder contains two more examples of using this implementation.

## Project structure
- `gcforest/` map contains the logic of the implementation,
- `data/` map contains a few data sets that were used to test this implementation,
- `examples/` map contains a few examples on how to use the implementation,
- `tests/` map constains tests (WIP).


## References
\[1\] Zhou, Z. H., and Feng, J. (2017).
Deep forest: Towards an alternative to deep neural networks.
In IJCAI-2017. 

\[2\] Ho, T. K. (1998). 
The random subspace method for constructing decision forests.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(8), 832-844.

\[3\] Zheng, Z. (2000).
Constructing X-of-N attributes for decision tree learning.
Machine learning, 40(1), 35-75.