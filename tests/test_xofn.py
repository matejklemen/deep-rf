import unittest
import numpy as np

from gcforest import xofn


class TestXOfN(unittest.TestCase):
    def test_gini(self):
        # clean data set
        self.assertEqual(xofn._gini([0, 0, 0, 5, 0, 0], 5), 0)
        # binary classes, 1 of each
        self.assertEqual(xofn._gini([1, 1], 2), 0.5)
        # multiple classes
        self.assertAlmostEqual(xofn._gini([2, 3, 2, 0, 5, 1], 13), 0.74556, places=5)

    def test_thresh_search(self):
        feat1 = np.array([1.0, -1.3, -1.3, 0.7, -1.3, 3.5, 1.0, 2.1, 2.1, 7.5, 3.5])
        lbl1 = np.array(["ClassA", "ClassA", "ClassB", "ClassB", "ClassC", "ClassC", "ClassA", "ClassC", "ClassA",
                         "ClassC", "ClassB"])
        uniq_thresh1 = np.array([-1.3, 0.7, 1.0, 2.1, 3.5, 7.5])
        best_gini, idx_best_thresh = xofn._res_gini_numerical(feat1, lbl1, uniq_thresh1)
        self.assertAlmostEqual(best_gini, 0.57576, places=5)
        self.assertEqual(uniq_thresh1[idx_best_thresh], 3.5)
