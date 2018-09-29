import unittest
import numpy as np

from gcforest.cascade_forest import EndingLayerAverage


class TestCascadeForest(unittest.TestCase):
    def test_ending_layer_shape(self):
        """
        - tests averaging for examples where a single vector consists of probabilities (in this case, these are not
        really probabilities) from 8 random forests for 3 classes (i.e. each vector is of length 8 * 3)
        """
        end_layer = EndingLayerAverage(classes_=np.array(["ClassA", "ClassB", "ClassC"]))
        test_feats = np.random.sample((10, 24))

        proba_preds = end_layer.predict_proba(test_feats)

        self.assertEqual(proba_preds.shape[0], 10)
        self.assertEqual(proba_preds.shape[1], 3)

    def test_average_values(self):
        """
        - tests if probabilities for same classes (for each example) are properly averaged
        """
        end_layer1 = EndingLayerAverage(classes_=np.array(["ClassA", "ClassB", "ClassC", "ClassD"]))
        end_layer2 = EndingLayerAverage(classes_=np.array(["ClassA", "ClassB"]))

        feats1 = np.array([[0.15, 0.2, 0.1, 0.55],
                          [1, 0, 0, 0],
                          [0.57, 0.23, 0.07, 0.13],
                          [0.01, 0.1, 0.8, 0.09],
                          [0.3, 0.3, 0.3, 0.1]])
        # each row: [p1(classA), p1(classB), p2(ClassA), p2(classB)]
        feats2 = np.array([[1, 0, 0.7, 0.3],
                           [0.01, 0.99, 0.22, 0.78],
                           [0.95, 0.05, 0.13, 0.87]])

        proba_preds1 = end_layer1.predict_proba(feats1)
        proba_preds2 = end_layer2.predict_proba(feats2)

        np.testing.assert_array_almost_equal(proba_preds1, feats1)
        np.testing.assert_array_almost_equal(proba_preds2, np.array([[0.85, 0.15],
                                                                    [0.115, 0.885],
                                                                    [0.54, 0.46]]))
