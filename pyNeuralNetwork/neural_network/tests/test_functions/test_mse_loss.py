import unittest
from unittest import TestCase

from neural_network.functions import MSELoss


class TestMSELoss(TestCase):
    """Tests the MSELoss class.
    """

    def setUp(self):
        self.loss = MSELoss()

    def test_construct(self):
        self.assertIsInstance(self.loss, MSELoss)

    def test_call(self):
        self.assertAlmostEqual(0.25, self.loss(0.8, 1.3),
                               places=8)
        self.assertAlmostEqual(0.0, self.loss(0.8, 0.8),
                               places=8)

    def test_gradient(self):
        self.assertAlmostEqual(-1.0, self.loss.gradient(0.8, 1.3),
                               places=8)
        self.assertAlmostEqual(0.0, self.loss.gradient(0.8, 0.8),
                               places=8)


if __name__ == '__main__':
    unittest.main()
