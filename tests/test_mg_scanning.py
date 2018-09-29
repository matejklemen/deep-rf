import unittest
import numpy as np

from gcforest.mg_scanning import Grain


class TestMGScanning(unittest.TestCase):
    def setUp(self):
        # single example - 3 rows, 5 columns, flattened
        self.sample_data_single = np.array([[0.7, 1.1, 3.0, 9.3, 1.0], [-2.3, 11.5, 9.6, 5.6, 1.6], [9.8, 1.0, 5.6, 0.7, 4.1]]).flatten()
        # two examples - each has 3 rows and 4 columns
        self.sample_data_multiple = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]])

    def test_1d_window_single(self):
        """
        - test 1D window of shape [1, 3] on a single example
        - test that slice_data method treats window sizes [1, 3] and 3 equally
        """
        grain1 = Grain(window_size=3, single_shape=[3, 5])
        grain2 = Grain(window_size=(1, 3), single_shape=[3, 5])
        desired_out = np.array([[0.7, 1.1, 3.0],
                                [1.1, 3.0, 9.3],
                                [3.0, 9.3, 1.0],
                                [-2.3, 11.5, 9.6],
                                [11.5, 9.6, 5.6],
                                [9.6, 5.6, 1.6],
                                [9.8, 1.0, 5.6],
                                [1.0, 5.6, 0.7],
                                [5.6, 0.7, 4.1]])

        np.testing.assert_array_almost_equal(grain1.slice_data(self.sample_data_single), desired_out)
        np.testing.assert_array_almost_equal(grain2.slice_data(self.sample_data_single), desired_out)

    def test_2d_window_single(self):
        """
        - test 2D window of shapes [3, 2], [3, 1] and [2, 2] on a single example
        - test a non-default stride ([1, 2]) on a single example
        """
        grain1 = Grain(window_size=[3, 2], single_shape=[3, 5])
        grain2 = Grain(window_size=[3, 1], single_shape=[3, 5])
        grain3 = Grain(window_size=[2, 2], single_shape=[3, 5])
        grain4 = Grain(window_size=[3, 2], single_shape=[3, 5], stride=[1, 2])

        desired_out1 = np.array([[0.7, 1.1, -2.3, 11.5, 9.8, 1.0],
                                 [1.1, 3.0, 11.5, 9.6, 1.0, 5.6],
                                 [3.0, 9.3, 9.6, 5.6, 5.6, 0.7],
                                 [9.3, 1.0, 5.6, 1.6, 0.7, 4.1]])
        desired_out2 = np.array([[0.7, -2.3, 9.8],
                                 [1.1, 11.5, 1.0],
                                 [3.0, 9.6, 5.6],
                                 [9.3, 5.6, 0.7],
                                 [1.0, 1.6, 4.1]])
        desired_out3 = np.array([[0.7, 1.1, -2.3, 11.5],
                                 [1.1, 3.0, 11.5, 9.6],
                                 [3.0, 9.3, 9.6, 5.6],
                                 [9.3, 1.0, 5.6, 1.6],
                                 [-2.3, 11.5, 9.8, 1.0],
                                 [11.5, 9.6, 1.0, 5.6],
                                 [9.6, 5.6, 5.6, 0.7],
                                 [5.6, 1.6, 0.7, 4.1]])
        desired_out4 = np.array([[0.7, 1.1, -2.3, 11.5, 9.8, 1.0],
                                 [3.0, 9.3, 9.6, 5.6, 5.6, 0.7]])

        np.testing.assert_array_almost_equal(grain1.slice_data(self.sample_data_single), desired_out1)
        np.testing.assert_array_almost_equal(grain2.slice_data(self.sample_data_single), desired_out2)
        np.testing.assert_array_almost_equal(grain3.slice_data(self.sample_data_single), desired_out3)
        np.testing.assert_array_almost_equal(grain4.slice_data(self.sample_data_single), desired_out4)

    def test_1d_window_multiple(self):
        """
        - test 1D window of shape [1, 3] on a "data set" (2 examples stacked)
        - test that slice_data method treats window sizes [1, 3] and 3 equally
        """
        grain1 = Grain(window_size=3, single_shape=[3, 4])
        grain2 = Grain(window_size=(1, 3), single_shape=[3, 4])

        desired_out = np.array([[1, 2, 3],
                                [2, 3, 4],
                                [5, 6, 7],
                                [6, 7, 8],
                                [9, 10, 11],
                                [10, 11, 12],
                                [101, 102, 103],
                                [102, 103, 104],
                                [105, 106, 107],
                                [106, 107, 108],
                                [109, 110, 111],
                                [110, 111, 112]])

        np.testing.assert_array_almost_equal(grain1.slice_data(self.sample_data_multiple), desired_out)
        np.testing.assert_array_almost_equal(grain2.slice_data(self.sample_data_multiple), desired_out)

    def test_2d_window_multiple(self):
        """
        - test 2D window of shapes [3, 2] and [2, 2] on a "data set" (2 examples stacked)
        """
        grain1 = Grain(window_size=[3, 2], single_shape=[3, 4])
        grain2 = Grain(window_size=[2, 2], single_shape=[3, 4])

        desired_out1 = np.array([[1, 2, 5, 6, 9, 10],
                                 [2, 3, 6, 7, 10, 11],
                                 [3, 4, 7, 8, 11, 12],
                                 [101, 102, 105, 106, 109, 110],
                                 [102, 103, 106, 107, 110, 111],
                                 [103, 104, 107, 108, 111, 112]])
        desired_out2 = np.array([[1, 2, 5, 6],
                                 [2, 3, 6, 7],
                                 [3, 4, 7, 8],
                                 [5, 6, 9, 10],
                                 [6, 7, 10, 11],
                                 [7, 8, 11, 12],
                                 [101, 102, 105, 106],
                                 [102, 103, 106, 107],
                                 [103, 104, 107, 108],
                                 [105, 106, 109, 110],
                                 [106, 107, 110, 111],
                                 [107, 108, 111, 112]])

        np.testing.assert_array_almost_equal(grain1.slice_data(self.sample_data_multiple), desired_out1)
        np.testing.assert_array_almost_equal(grain2.slice_data(self.sample_data_multiple), desired_out2)