import unittest
from unittest import TestCase

from neural_network.functions import CrossEntropyLoss


class TestCrossEntropyLoss(TestCase):
    """Tests the `CrossEntropyLoss` class
    """

    def setUp(self):
        self.loss = CrossEntropyLoss()
        self.y_hat = [-1, 0.4, 0.2, 2]

    def test_construct(self):
        self.assertIsInstance(self.loss, CrossEntropyLoss)

    def test_call_erroneous(self):
        with self.assertRaises(ValueError) as ve:
            self.loss(self.y_hat, y=0)
        self.assertEqual(str(ve.exception), "Softmax value should be between"
                                            " 0 and 1 (y_hat[0] = -1)")

    def test_call(self):
        self.assertAlmostEqual(0.91629073, self.loss(self.y_hat, y=1),
                               places=8)


if __name__ == '__main__':
    unittest.main()
