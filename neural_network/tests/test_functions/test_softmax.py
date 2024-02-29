import unittest
from unittest import TestCase

from neural_network import Softmax


class TestSoftmax(TestCase):
    """Tests the `Softmax` class
    """

    def setUp(self):
        self.softmax = Softmax()
        self.z_list = [-2.0, 0.0, 2.0, 4.0]

    def test_construct(self):
        self.assertEqual(1.0, self.softmax._normalisation)
        self.assertEqual(0.0, self.softmax._max_z)

    def test_normalise(self):
        self.softmax.normalise(self.z_list)
        self.assertEqual(4.0, self.softmax._max_z)
        self.assertAlmostEqual(1.15612967, self.softmax._normalisation,
                               places=8)

    def test_call(self):
        self.softmax.normalise(self.z_list)
        expected_softmax = [0.00214401, 0.01584220, 0.11705891, 0.86495488]
        for i in range(4):
            self.assertAlmostEqual(expected_softmax[i],
                                   self.softmax(self.z_list[i]), places=8)


if __name__ == '__main__':
    unittest.main()
