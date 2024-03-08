import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network import NormalDataGenerator


class TestNormalDataGenerator(TestCase):
    """Tests the NormalDataGenerator class
    """

    one_coord_normal_data = [np.array([0.8, 1.2, 1.1, 0.6, -0.1])]
    two_coord_normal_data = [np.array([-0.8, -0.5, 1.4, 0.5, 0.2]),
                             np.array([0.5, 2.1, -0.5, 4.2, 3.1])]
    str_classifier_data = [np.array([4.5, 6.2, 11.7, 7.6, 8.8])]

    @staticmethod
    def one_coord_classifier(x_1: float) -> int:
        if x_1 < 0:
            return 3
        if 0 < x_1 < 1:
            return 4
        return 5

    @staticmethod
    def two_coords_classifier(x_1: float, x_2: float) -> int:
        return int(x_1 + x_2 ** 2 > 1)

    @staticmethod
    def string_classifier(x_1: float) -> str:
        int_x = int(x_1)
        if int_x % 2 == 0:
            return "Even"
        else:
            return "Odd"

    def setUp(self):
        self.one_coord_gen = NormalDataGenerator(self.one_coord_classifier,
                                                 num_datapoints=5,
                                                 means=[1.0], std_devs=[1.0])
        self.two_coord_gen = NormalDataGenerator(self.two_coords_classifier,
                                                 num_datapoints=5,
                                                 means=[0.0, 2.0],
                                                 std_devs=[1.0, 2.0])
        self.str_gen = NormalDataGenerator(self.string_classifier,
                                           num_datapoints=5, means=[8.0],
                                           std_devs=[3.0])

    def test_construct_erroneous(self):
        with self.assertRaises(ValueError) as ve_1:
            NormalDataGenerator(self.one_coord_classifier, num_datapoints=10,
                                means=[1.0, 2.0], std_devs=[1.0])
        self.assertEqual("The function method accepts 1 parameters but "
                         "we have 2 means.", str(ve_1.exception))
        with self.assertRaises(ValueError) as ve_2:
            NormalDataGenerator(self.one_coord_classifier, num_datapoints=10,
                                means=[1.0], std_devs=[1.0, 2.0])
        self.assertEqual("The function method accepts 1 parameters but "
                         "we have 2 standard deviations.", str(ve_2.exception))
        with self.assertRaises(ValueError) as ve_3:
            NormalDataGenerator(self.two_coords_classifier, num_datapoints=10,
                                means=[1.0, 2.0], std_devs=[1.0, 0.0])
        self.assertEqual("All standard deviations must be positive "
                         "(0.0 <= 0)", str(ve_3.exception))

    def test_construct(self):
        self.assertEqual(self.one_coord_classifier,
                         self.one_coord_gen._function)
        self.assertListEqual([1.0], self.one_coord_gen._means)
        self.assertListEqual([1.0], self.one_coord_gen._std_devs)
        self.assertEqual(self.two_coords_classifier,
                         self.two_coord_gen._function)
        self.assertListEqual([0.0, 2.0], self.two_coord_gen._means)
        self.assertListEqual([1.0, 2.0], self.two_coord_gen._std_devs)

    @mock.patch('numpy.random.normal',
                side_effect=one_coord_normal_data)
    def test_generate_data_one_coord(self, mock_normal):
        self.one_coord_gen._generate_data()
        mock_normal.assert_called_once_with(loc=1.0, scale=1.0, size=5)
        self.assertListEqual(self.one_coord_normal_data,
                             self.one_coord_gen._x)

    @mock.patch('numpy.random.normal',
                side_effect=two_coord_normal_data)
    def test_generate_data_two_coord(self, mock_normal):
        self.two_coord_gen._generate_data()
        normal_calls = mock_normal.call_args_list
        self.assertListEqual([mock.call(loc=0.0, scale=1.0, size=5),
                              mock.call(loc=2.0, scale=2.0, size=5)],
                             normal_calls)
        self.assertEqual(2, mock_normal.call_count)
        self.assertListEqual(self.two_coord_normal_data,
                             self.two_coord_gen._x)

    @mock.patch('neural_network.data_generators.normal_data_generator'
                '.NormalDataGenerator._generate_data')
    def test_call_one_coord(self, mock_gen_method):
        # This represents what _generate_data would do
        self.one_coord_gen._x = self.one_coord_normal_data
        actual_df = self.one_coord_gen()
        mock_gen_method.assert_called_once()
        expected_df = pd.DataFrame(columns=['x_1', 'y'])
        expected_df['x_1'] = self.one_coord_normal_data[0]

        # These are the classes each datapoint is assigned to using the
        # classifier method
        expected_df['y'] = np.array([4, 5, 5, 4, 3], dtype=object)
        pd.testing.assert_frame_equal(expected_df, actual_df)

    @mock.patch('neural_network.data_generators.normal_data_generator'
                '.NormalDataGenerator._generate_data')
    def test_call_two_coord(self, mock_gen_method):
        # This represents what _generate_data would do
        self.two_coord_gen._x = self.two_coord_normal_data
        actual_df = self.two_coord_gen()
        mock_gen_method.assert_called_once()
        expected_df = pd.DataFrame(columns=['x_1', 'x_2', 'y'])
        expected_df['x_1'] = self.two_coord_normal_data[0]
        expected_df['x_2'] = self.two_coord_normal_data[1]

        # These are the classes each datapoint is assigned to using the
        # classifier method
        expected_df['y'] = np.array([0, 1, 1, 1, 1], dtype=object)
        pd.testing.assert_frame_equal(expected_df, actual_df)

    @mock.patch('neural_network.data_generators.normal_data_generator'
                '.NormalDataGenerator._generate_data')
    def test_call_str(self, mock_gen_method):
        # This represents what _generate_data would do
        self.str_gen._x = self.str_classifier_data
        actual_df = self.str_gen()
        mock_gen_method.assert_called_once()
        expected_df = pd.DataFrame(columns=['x_1', 'y'])
        expected_df['x_1'] = self.str_classifier_data[0]

        # These are the classes each datapoint is assigned to using the
        # classifier method
        expected_df['y'] = ["Even", "Even", "Odd", "Odd", "Even"]
        pd.testing.assert_frame_equal(expected_df, actual_df)


if __name__ == '__main__':
    unittest.main()
