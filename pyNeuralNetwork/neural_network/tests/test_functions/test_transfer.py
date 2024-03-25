import unittest
from unittest import TestCase

from neural_network import TransferFunction


class TestTransferFunction(TestCase):
    """Tests the `TransferFunction` class
    """

    def setUp(self):
        self.transfer = TransferFunction()
        self.o_list = [1.0, 2.0, 3.0]
        self.w_list = [-1.0, 0.0, 1.0, 2.0]

    def test_construct(self):
        self.assertIsInstance(self.transfer, TransferFunction)

    def test_call_erroneous(self):
        with self.assertRaises(ValueError) as ve_1:
            self.transfer([1, 2, 3])
        self.assertEqual(str(ve_1.exception), "w cannot be None")

        with self.assertRaises(ValueError) as ve_2:
            self.transfer([1, 2, 3], [4, 5, 6])
        self.assertEqual(str(ve_2.exception), "w is not one element longer "
                                              "than o (3 != 3 + 1)")

    def test_call(self):
        result = self.transfer(self.o_list, self.w_list)

        # 1 * -1 + 2 * 0 + 3 * 1 + 2
        self.assertEqual(4.0, result)

    def test_gradient(self):
        self.assertListEqual([1.0, 2.0, 3.0, 0.0],
                             self.transfer.gradient(self.o_list))
        self.assertListEqual([1.0, 2.0, 3.0, 0.0],
                             self.transfer.gradient(self.o_list, self.w_list))


if __name__ == '__main__':
    unittest.main()
