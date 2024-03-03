import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network import UniformDataGenerator


class TestUniformDataGenerator(TestCase):
    """Tests the UniformDataGenerator class
    """

    one_coord_uniform_data = [np.array([1.8, 1.2, 1.1, 1.6, 1.1])]
    two_coord_uniform_data = [np.array([0.8, 0.5, 0.4, 0.5, 0.2]),
                             np.array([0.2, 2.1, 3.5, 4.2, 3.1])]
    str_classifier_data = [np.array([9.5, 14.2, 11.7, 13.6, 8.8])]

    @staticmethod
    def one_coord_classifier(x_1: float) -> int:
        if x_1 < 0:
            return 3
        if 0 < x_1 < 1:
            return 4
        return 5

    @staticmethod
    def two_coords_classifier(x_1: float, x_2: float) -> int:
        return int(x_1 - x_2 ** 2 > 0)

    @staticmethod
    def string_classifier(x_1: float) -> str:
        int_x = int(x_1)
        if int_x % 2 == 0:
            return "Even"
        else:
            return "Odd"

    def setUp(self):
        self.one_coord_gen = UniformDataGenerator(self.one_coord_classifier,
                                                  num_datapoints=5,
                                                  lower_bounds=[1.0],
                                                  upper_bounds=[2.0])
        self.two_coord_gen = UniformDataGenerator(self.two_coords_classifier,
                                                  num_datapoints=5,
                                                  lower_bounds=[0.0, 0.0],
                                                  upper_bounds=[1.0, 5.0])
        self.str_gen = UniformDataGenerator(self.string_classifier,
                                            num_datapoints=5,
                                            lower_bounds=[8.0],
                                            upper_bounds=[15.0])

    def test_construct_erroneous(self):
        with self.assertRaises(ValueError) as ve_1:
            UniformDataGenerator(self.one_coord_classifier, num_datapoints=10,
                                 lower_bounds=[1.0, 2.0], upper_bounds=[1.0])
        self.assertEqual("The classifier method accepts 1 parameters but "
                         "we have 2 lower bounds.", str(ve_1.exception))
        with self.assertRaises(ValueError) as ve_2:
            UniformDataGenerator(self.one_coord_classifier, num_datapoints=10,
                                 lower_bounds=[1.0], upper_bounds=[1.0, 2.0])
        self.assertEqual("The classifier method accepts 1 parameters but "
                         "we have 2 upper bounds.", str(ve_2.exception))
        with self.assertRaises(ValueError) as ve_3:
            UniformDataGenerator(self.two_coords_classifier, num_datapoints=10,
                                 lower_bounds=[1.0, -1.0],
                                 upper_bounds=[1.0, 0.0])
        self.assertEqual("All lower bounds must be lower than their "
                         "related upper bounds (1.0 >= 1.0)",
                         str(ve_3.exception))

    def test_construct(self):
        self.assertEqual(self.one_coord_classifier,
                         self.one_coord_gen._classifier)
        self.assertListEqual([1.0], self.one_coord_gen._lower_bounds)
        self.assertListEqual([2.0], self.one_coord_gen._upper_bounds)
        self.assertEqual(self.two_coords_classifier,
                         self.two_coord_gen._classifier)
        self.assertListEqual([0.0, 0.0], self.two_coord_gen._lower_bounds)
        self.assertListEqual([1.0, 5.0], self.two_coord_gen._upper_bounds)

    @mock.patch('numpy.random.uniform',
                side_effect=one_coord_uniform_data)
    def test_generate_data_one_coord(self, mock_normal):
        self.one_coord_gen._generate_data()
        mock_normal.assert_called_once_with(low=1.0, high=2.0, size=5)
        self.assertListEqual(self.one_coord_uniform_data,
                             self.one_coord_gen._x)

    @mock.patch('numpy.random.uniform',
                side_effect=two_coord_uniform_data)
    def test_generate_data_two_coord(self, mock_normal):
        self.two_coord_gen._generate_data()
        normal_calls = mock_normal.call_args_list
        self.assertListEqual([mock.call(low=0.0, high=1.0, size=5),
                              mock.call(low=0.0, high=5.0, size=5)],
                             normal_calls)
        self.assertEqual(2, mock_normal.call_count)
        self.assertListEqual(self.two_coord_uniform_data,
                             self.two_coord_gen._x)

    @mock.patch('neural_network.data_generators.uniform_data_generator'
                '.UniformDataGenerator._generate_data')
    def test_call_one_coord(self, mock_gen_method):
        # This represents what _generate_data would do
        self.one_coord_gen._x = self.one_coord_uniform_data
        actual_df, categories = self.one_coord_gen()
        mock_gen_method.assert_called_once()
        expected_df = pd.DataFrame(columns=['x_1', 'y'])
        expected_df['x_1'] = self.one_coord_uniform_data[0]

        # These are the classes each datapoint is assigned to using the
        # classifier method and renaming the indices to be 0-2
        expected_df['y'] = np.array([0, 0, 0, 0, 0])
        pd.testing.assert_frame_equal(expected_df, actual_df)
        self.assertListEqual([5], categories)

    @mock.patch('neural_network.data_generators.uniform_data_generator'
                '.UniformDataGenerator._generate_data')
    def test_call_two_coord(self, mock_gen_method):
        # This represents what _generate_data would do
        self.two_coord_gen._x = self.two_coord_uniform_data
        actual_df, categories = self.two_coord_gen()
        mock_gen_method.assert_called_once()
        expected_df = pd.DataFrame(columns=['x_1', 'x_2', 'y'])
        expected_df['x_1'] = self.two_coord_uniform_data[0]
        expected_df['x_2'] = self.two_coord_uniform_data[1]

        # These are the classes each datapoint is assigned to using the
        # classifier method
        expected_df['y'] = np.array([1, 0, 0, 0, 0])
        pd.testing.assert_frame_equal(expected_df, actual_df)
        self.assertListEqual([0, 1], categories)

    @mock.patch('neural_network.data_generators.uniform_data_generator'
                '.UniformDataGenerator._generate_data')
    def test_call_str(self, mock_gen_method):
        # This represents what _generate_data would do
        self.str_gen._x = self.str_classifier_data
        actual_df, categories = self.str_gen()
        mock_gen_method.assert_called_once()
        expected_df = pd.DataFrame(columns=['x_1', 'y'])
        expected_df['x_1'] = self.str_classifier_data[0]

        # These are the classes each datapoint is assigned to using the
        # classifier method
        expected_df['y'] = np.array([1, 0, 1, 1, 0])
        pd.testing.assert_frame_equal(expected_df, actual_df)
        self.assertListEqual(["Even", "Odd"], categories)


if __name__ == '__main__':
    unittest.main()
