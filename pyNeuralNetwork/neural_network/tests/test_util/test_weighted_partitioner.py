import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network.util import WeightedPartitioner


class TestWeightedPartitioner(TestCase):
    """Tests the `WeightedPartitioner` class
    """
    mock_random_choice = [3, 4, 1, 8, 3, 6, 6, 9, 2, 5]
    mock_random_choices = [2, 0, 1, 0, 2, 0, 0, 1, 2, 1]

    def setUp(self):
        self.ints = list(range(10))
        self.data = np.array([[3, 2, 5, 1],
                              [6, -2, -3, 1],
                              [0, 1, 0, 2],
                              [-4, -3, -2, 2],
                              [1, -9, 2, 0],
                              [2, 4, -3, 1],
                              [-4, -2, 5, 0],
                              [2, 3, 1, 1],
                              [-9, -3, 2, 0],
                              [2, 3, -4, 1]])
        self.df = pd.DataFrame(self.data, columns=["x_1", "x_2", "x_3", "y"])
        self.even_partitioner = WeightedPartitioner(10, 5, self.df)
        self.uneven_partitioner = WeightedPartitioner(10, 3, self.df)
        self.big_partitioner = WeightedPartitioner(10, 10, self.df)
        self.small_partitioner = WeightedPartitioner(10, 1, self.df)
        self.reg_data = np.array([[3, 2, 5, 1.4],
                                  [6, -2, -3, 6.2],
                                  [0, 1, 0, 5.3],
                                  [-4, -3, -2, 8.8],
                                  [1, -9, 2, 3.4],
                                  [2, 4, -3, 3.1],
                                  [-4, -2, 5, 2.6],
                                  [2, 3, 1, 1.0],
                                  [-9, -3, 2, 9.0],
                                  [2, 3, -4, 8.2]])
        self.reg_df = pd.DataFrame(self.reg_data, columns=["x_1", "x_2",
                                                           "x_3", "y"])
        self.reg_partitioner = WeightedPartitioner(10, 5, self.reg_df,
                                                   do_regression=True,
                                                   num_bins=8)

    def test_construct_erroneous(self):
        with self.assertRaises(ValueError) as ve:
            WeightedPartitioner(8, 5, self.df)
        self.assertEqual("n must equal the length of the dataframe "
                         "(n = 8, len(df) = 10)",
                         str(ve.exception))

    def test_construct(self):
        self.assertEqual(10, self.even_partitioner._n)
        self.assertEqual(5, self.even_partitioner._m)
        self.assertEqual(3, self.even_partitioner._num_bins)
        self.assertDictEqual({0: [4, 6, 8], 1: [0, 1, 5, 7, 9], 2: [2, 3]},
                             self.even_partitioner._class_dict)

    def test_construct_regression(self):
        self.assertEqual(10, self.reg_partitioner._n)
        self.assertEqual(5, self.reg_partitioner._m)
        self.assertEqual(6, self.reg_partitioner._num_bins)
        self.assertDictEqual({0: [0, 7], 1: [6], 2: [4, 5], 3: [2], 4: [1],
                              5: [3, 8, 9]},
                             self.reg_partitioner._class_dict)

    @mock.patch('random.choice',
                side_effect=mock_random_choice)
    @mock.patch('random.choices')
    def test_call_even_partitioner(self, mock_choices, mock_choice):
        mock_choices.return_value = self.mock_random_choices
        self.assertEqual([[3, 4, 1, 8, 3], [6, 6, 9, 2, 5]],
                         self.even_partitioner())
        mock_choices.assert_called_once_with(population=range(3), k=10)

        choice_calls = mock_choice.call_args_list
        expected_calls = [mock.call([2, 3]), mock.call([4, 6, 8]),
                          mock.call([0, 1, 5, 7, 9]), mock.call([4, 6, 8]),
                          mock.call([2, 3]), mock.call([4, 6, 8]),
                          mock.call([4, 6, 8]), mock.call([0, 1, 5, 7, 9]),
                          mock.call([2, 3]), mock.call([0, 1, 5, 7, 9])]
        self.assertListEqual(expected_calls, choice_calls)
        self.assertEqual(10, mock_choice.call_count)

    @mock.patch('random.choice',
                side_effect=mock_random_choice)
    @mock.patch('random.choices')
    def test_call_uneven_partitioner(self, mock_choices, mock_choice):
        mock_choices.return_value = self.mock_random_choices
        self.assertEqual([[3, 4, 1], [8, 3, 6], [6, 9, 2], [5]],
                         self.uneven_partitioner())

    @mock.patch('random.choice',
                side_effect=mock_random_choice)
    @mock.patch('random.choices')
    def test_call_big_partitioner(self, mock_choices, mock_choice):
        mock_choices.return_value = self.mock_random_choices
        self.assertEqual([[3, 4, 1, 8, 3, 6, 6, 9, 2, 5]],
                         self.big_partitioner())

    @mock.patch('random.choice',
                side_effect=mock_random_choice)
    @mock.patch('random.choices')
    def test_call_small_partitioner(self, mock_choices, mock_choice):
        mock_choices.return_value = self.mock_random_choices
        self.assertEqual([[3], [4], [1], [8], [3],
                          [6], [6], [9], [2], [5]],
                         self.small_partitioner())


if __name__ == '__main__':
    unittest.main()
