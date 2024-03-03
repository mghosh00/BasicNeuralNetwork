import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network import DataSplitter


class TestDataSplitter(TestCase):
    """Tests the `DataSplitter` class
    """

    @mock.patch('pandas.read_csv')
    def setUp(self, mock_read):
        self.data = np.array([[3, 2, 5, 1],
                              [6, -2, -3, 1],
                              [0, 1, 0, 1],
                              [-4, -3, -2, 0],
                              [1, -9, 2, 0],
                              [2, 4, -3, 1],
                              [-4, -2, 5, 0],
                              [2, 3, 1, 1],
                              [-9, -3, 2, 0],
                              [2, 3, -4, 1]])
        self.df = pd.DataFrame(self.data, columns=["a", "b", "c", "class"])
        mock_read.return_value = self.df
        self.splitter1 = DataSplitter('mock_path', [8, 1, 1])
        self.splitter2 = DataSplitter('mock_path', [80, 20])
        self.splitter3 = DataSplitter('mock_path', [12, 2, 2])
        self.splitter4 = DataSplitter('mock_path', [7])
        self.splitter5 = DataSplitter('mock_path', [14, 1, 1])

    @mock.patch('pandas.read_csv')
    def test_construct_erroneous(self, mock_read):
        with self.assertRaises(ValueError) as ve_1:
            DataSplitter('mock_path', [])
        self.assertEqual("proportions must have 1-3 elements denoting the"
                         "train:validation:test ratio", str(ve_1.exception))
        with self.assertRaises(ValueError) as ve_2:
            DataSplitter('mock_path', [3, 4, 2, 5])
        self.assertEqual("proportions must have 1-3 elements denoting the"
                         "train:validation:test ratio", str(ve_2.exception))

    def test_construct(self):
        pd.testing.assert_frame_equal(self.df, self.splitter1._df)
        self.assertListEqual([8, 1, 1], self.splitter1._proportions)

    def test_split_1(self):
        # Tests an even split with training, validation and testing
        expected_dfs = (self.df.iloc[:8], self.df.iloc[8:9],
                        self.df.iloc[9:])
        expected_dfs[1].index = [0]
        expected_dfs[2].index = [0]
        actual_dfs = self.splitter1.split()
        for i in range(3):
            pd.testing.assert_frame_equal(expected_dfs[i], actual_dfs[i])

    def test_split_2(self):
        # Tests an even split with training and validation
        expected_dfs = (self.df.iloc[:8], self.df.iloc[8:])
        expected_dfs[1].index = [0, 1]
        actual_dfs = self.splitter2.split()
        for i in range(2):
            pd.testing.assert_frame_equal(expected_dfs[i], actual_dfs[i])

    def test_split_3(self):
        # Tests an uneven split with training, validation and testing
        expected_dfs = (self.df.iloc[:7], self.df.iloc[7:8],
                        self.df.iloc[8:])
        expected_dfs[1].index = [0]
        expected_dfs[2].index = [0, 1]
        actual_dfs = self.splitter3.split()
        for i in range(3):
            pd.testing.assert_frame_equal(expected_dfs[i], actual_dfs[i])

    def test_split_4(self):
        # Tests with just a training set
        expected_dfs = tuple([self.df])
        actual_dfs = self.splitter4.split()
        pd.testing.assert_frame_equal(expected_dfs[0], actual_dfs[0])

    def test_split_5(self):
        # Tests that we cannot get dfs of size 0
        expected_dfs = (self.df.iloc[:8], self.df.iloc[8:9],
                        self.df.iloc[9:])
        expected_dfs[1].index = [0]
        expected_dfs[2].index = [0]
        actual_dfs = self.splitter5.split()
        for i in range(3):
            pd.testing.assert_frame_equal(expected_dfs[i], actual_dfs[i])


if __name__ == '__main__':
    unittest.main()
