import unittest
from unittest import TestCase
import random

from neural_network import Partitioner


class TestPartitioner(TestCase):
    """Tests the `Partitioner` class
    """

    def setUp(self):
        random.seed(42)
        self.ints = list(range(10))
        # Below is the shuffle for seed = 42
        self.shuffled_ints = [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
        self.even_partitioner = Partitioner(10, 5)
        self.uneven_partitioner = Partitioner(10, 3)
        self.big_partitioner = Partitioner(10, 10)
        self.small_partitioner = Partitioner(10, 1)

    def test_construct_erroneous(self):
        with self.assertRaises(ValueError) as ve_1:
            Partitioner(1, 0)
        self.assertEqual("n (1) and m (0) must be positive integers",
                         str(ve_1.exception))
        with self.assertRaises(ValueError) as ve_2:
            Partitioner(4, 5)
        self.assertEqual("m (5) cannot be greater than n (4)",
                         str(ve_2.exception))

    def test_construct(self):
        self.assertEqual(10, self.even_partitioner._n)
        self.assertEqual(5, self.even_partitioner._m)

    def test_call_even_partitioner(self):
        self.assertEqual([[7, 3, 2, 8, 5], [6, 9, 4, 0, 1]],
                         self.even_partitioner())

    def test_call_uneven_partitioner(self):
        self.assertEqual([[7, 3, 2], [8, 5, 6], [9, 4, 0], [1]],
                         self.uneven_partitioner())

    def test_call_big_partitioner(self):
        self.assertEqual([[7, 3, 2, 8, 5, 6, 9, 4, 0, 1]],
                         self.big_partitioner())

    def test_call_small_partitioner(self):
        self.assertEqual([[7], [3], [2], [8], [5],
                          [6], [9], [4], [0], [1]],
                         self.small_partitioner())


if __name__ == '__main__':
    unittest.main()
