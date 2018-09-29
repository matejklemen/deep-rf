import unittest
from warnings import warn

from gcforest import datasets


class TestDatasets(unittest.TestCase):
    def test_yeast(self):
        train_X_yeast, train_y_yeast, test_X_yeast, test_y_yeast = datasets.prep_yeast()
        self._check_shapes([train_X_yeast.shape, train_y_yeast.shape, test_X_yeast.shape, test_y_yeast.shape],
                           [(1038, 8), (1038,), (446, 8), (446,)])

    def test_adult(self):
        train_X_adult, train_y_adult, test_X_adult, test_y_adult = datasets.prep_adult()
        self._check_shapes([train_X_adult.shape, train_y_adult.shape, test_X_adult.shape, test_y_adult.shape],
                           [(32561, 113), (32561,), (16281, 113), (16281,)])

    def test_letter(self):
        train_X_letter, train_y_letter, test_X_letter, test_y_letter = datasets.prep_letter()
        self._check_shapes([train_X_letter.shape, train_y_letter.shape, test_X_letter.shape, test_y_letter.shape],
                           [(16000, 16), (16000,), (4000, 16), (4000,)])

    def test_mnist(self):
        try:
            import keras.datasets
        except ImportError:
            warn("MNIST data set test skipped due to dependency missing...")
            return

        train_X_mnist, train_y_mnist, test_X_mnist, test_y_mnist = datasets.prep_mnist_org_paper(mode="small")
        self._check_shapes([train_X_mnist.shape, train_y_mnist.shape, test_X_mnist.shape, test_y_mnist.shape],
                           [(2000, 28 * 28), (2000,), (1000, 28 * 28), (1000,)])

        train_X_mnist, train_y_mnist, test_X_mnist, test_y_mnist = datasets.prep_mnist_org_paper(mode="medium")
        self._check_shapes([train_X_mnist.shape, train_y_mnist.shape, test_X_mnist.shape, test_y_mnist.shape],
                           [(10000, 28 * 28), (10000,), (5000, 28 * 28), (5000,)])

    def test_orl(self):
        # test 5/7/9 images per person in training set
        train_X_orl, train_y_orl, test_X_orl, test_y_orl = datasets.prep_orl(train_imgs_person=5)
        self._check_shapes([train_X_orl.shape, train_y_orl.shape, test_X_orl.shape, test_y_orl.shape],
                           [(200, 64 * 64), (200,), (200, 64 * 64), (200,)])

        train_X_orl, train_y_orl, test_X_orl, test_y_orl = datasets.prep_orl(train_imgs_person=7)
        self._check_shapes([train_X_orl.shape, train_y_orl.shape, test_X_orl.shape, test_y_orl.shape],
                           [(280, 64 * 64), (280,), (120, 64 * 64), (120,)])

        train_X_orl, train_y_orl, test_X_orl, test_y_orl = datasets.prep_orl(train_imgs_person=9)
        self._check_shapes([train_X_orl.shape, train_y_orl.shape, test_X_orl.shape, test_y_orl.shape],
                           [(360, 64 * 64), (360,), (40, 64 * 64), (40,)])

    def _check_shapes(self, shapes, actual_shapes):
        shape_train_X, shape_train_y, shape_test_X, shape_test_y = shapes
        true_train_X, true_train_y, true_test_X, true_test_y = actual_shapes
        # `shapes` contains list of tuples, representing shapes train_X.shape, train_y.shape, test_X.shape, test_y.shape
        self.assertTupleEqual(shape_train_X, true_train_X)
        self.assertTupleEqual(shape_train_y, true_train_y)
        self.assertTupleEqual(shape_test_X, true_test_X)
        self.assertTupleEqual(shape_test_y, true_test_y)
