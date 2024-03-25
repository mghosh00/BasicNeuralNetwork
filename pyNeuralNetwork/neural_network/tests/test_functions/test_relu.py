import unittest
from unittest import TestCase

from neural_network.functions import ReLU


class TestReLU(TestCase):
    """Tests the `ReLU` class
    """

    def setUp(self):
        self.default_relu = ReLU()
        self.leaky_relu = ReLU(leak=0.01)

    def test_construct(self):
        self.assertEqual(0.0, self.default_relu._leak)
        self.assertEqual(0.01, self.leaky_relu._leak)

    def test_call(self):
        self.assertEqual(2.0, self.default_relu(2.0))
        self.assertEqual(0.0, self.default_relu(-1.0, [0.2, 0.4]))
        self.assertEqual(2.0, self.leaky_relu(2.0))
        self.assertEqual(-0.05, self.leaky_relu(-5.0))

    def test_gradient(self):
        self.assertEqual(1.0, self.default_relu.gradient(2.0))
        self.assertEqual(0.0, self.default_relu.gradient(-1.0))
        self.assertEqual(1.0, self.default_relu.gradient(0.0))
        self.assertEqual(1.0, self.leaky_relu.gradient(2.0,
                                                       w=[0.2, 0.5]))
        self.assertEqual(0.01, self.leaky_relu.gradient(-5.0))
        self.assertEqual(1.0, self.leaky_relu.gradient(0.0))


if __name__ == '__main__':
    unittest.main()
