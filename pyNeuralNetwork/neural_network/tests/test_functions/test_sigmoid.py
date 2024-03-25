import unittest
from unittest import TestCase

from neural_network.functions import Sigmoid


class TestSigmoid(TestCase):
    """Tests the `Sigmoid` class
    """

    def setUp(self):
        self.sigmoid = Sigmoid()

    def test_construct(self):
        self.assertIsInstance(self.sigmoid, Sigmoid)

    def test_call(self):
        self.assertEqual(0.5, self.sigmoid(0.0))
        self.assertEqual(0.5, self.sigmoid(0.0, [0.1, 0.2]))
        self.assertAlmostEqual(0.73105858, self.sigmoid(1.0),
                               places=8)
        self.assertAlmostEqual(0.26894142, self.sigmoid(-1.0),
                               places=8)

    def test_gradient(self):
        self.assertEqual(0.25, self.sigmoid.gradient(0.0))
        self.assertEqual(0.25, self.sigmoid.gradient(0.0, w=[0.1, 0.2]))
        self.assertAlmostEqual(0.19661193, self.sigmoid.gradient(1.0),
                               places=8)
        self.assertAlmostEqual(0.19661193, self.sigmoid.gradient(-1.0),
                               places=8)


if __name__ == '__main__':
    unittest.main()
