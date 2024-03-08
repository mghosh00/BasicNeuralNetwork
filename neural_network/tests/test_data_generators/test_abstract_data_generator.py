from typing import Union, Callable, Any
import unittest
from unittest import TestCase
from unittest import mock

import pandas as pd

from neural_network import AbstractDataGenerator


class TestAbstractDataGenerator(TestCase):
    """Tests the AbstractDataGenerator class
    """

    @staticmethod
    def erroneous_classifier() -> int:
        return 4

    @staticmethod
    def one_coord_classifier(x_1: float) -> int:
        if x_1 < 0:
            return 0
        if 0 < x_1 < 1:
            return 1
        return 2

    @staticmethod
    def two_coords_classifier(x_1: float, x_2: float) -> int:
        return int(x_1 + x_2 ** 2 > 1)

    def setUp(self):
        self.one_coord_gen = AbstractDataGenerator(self.one_coord_classifier,
                                                   num_datapoints=100)
        self.two_coord_gen = AbstractDataGenerator(self.two_coords_classifier,
                                                   num_datapoints=200)

    def test_construct_erroneous(self):
        with self.assertRaises(ValueError) as ve_1:
            AbstractDataGenerator(self.erroneous_classifier, num_datapoints=10)
        self.assertEqual("function must have at least one coordinate "
                         "(num_coordinates = 0)", str(ve_1.exception))
        with self.assertRaises(ValueError) as ve_2:
            AbstractDataGenerator(self.one_coord_classifier, num_datapoints=0)
        self.assertEqual("Must have at least one datapoint (num_datapoints"
                         " = 0)", str(ve_2.exception))

    def test_construct(self):
        self.assertEqual(self.one_coord_classifier,
                         self.one_coord_gen._function)
        self.assertEqual(1, self.one_coord_gen._dimensions)
        self.assertEqual(100, self.one_coord_gen._num_datapoints)
        pd.testing.assert_frame_equal(pd.DataFrame(columns=['x_1', 'y']),
                                      self.one_coord_gen._df)
        self.assertListEqual([], self.one_coord_gen._x)
        self.assertEqual(self.two_coords_classifier,
                         self.two_coord_gen._function)
        self.assertEqual(2, self.two_coord_gen._dimensions)
        self.assertEqual(200, self.two_coord_gen._num_datapoints)
        pd.testing.assert_frame_equal(pd.DataFrame(columns=['x_1', 'x_2',
                                                            'y']),
                                      self.two_coord_gen._df)
        self.assertListEqual([], self.two_coord_gen._x)
        self.assertEqual(Union[Callable[[float], Any],
                               Callable[[float, float], Any],
                               Callable[[float, float, float], Any],
                               Callable[[float, float, float, float], Any]],
                         self.one_coord_gen.custom_type)

    def test_generate_data(self):
        with self.assertRaises(NotImplementedError) as ve:
            self.one_coord_gen._generate_data()
        self.assertEqual("Cannot call _generate_data or __call__ from base "
                         "class", str(ve.exception))

    def test_call(self):
        with self.assertRaises(NotImplementedError) as ve:
            self.one_coord_gen()
        self.assertEqual("Cannot call _generate_data or __call__ from base "
                         "class", str(ve.exception))

    @mock.patch('pandas.DataFrame.to_csv')
    def test_write_to_csv(self, mock_write):
        # Default
        self.one_coord_gen.write_to_csv("test_title")
        # Non-default
        self.one_coord_gen.write_to_csv("test_title", "test_dir")
        write_calls = mock_write.call_args_list
        self.assertEqual(mock.call("test_title.csv"), write_calls[0])
        self.assertEqual(mock.call("test_dir/test_title.csv"), write_calls[1])
        self.assertEqual(2, mock_write.call_count)


if __name__ == '__main__':
    unittest.main()
