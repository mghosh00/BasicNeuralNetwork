import unittest
from unittest import TestCase

from neural_network.functions import AbstractFunction


class TestAbstractFunction(TestCase):
    """Tests the `AbstractFunction` class
    """

    def setUp(self):
        self.abstract_function = AbstractFunction()

    def test_construct(self):
        self.assertIsInstance(self.abstract_function, AbstractFunction)

    def test_call(self):
        with self.assertRaises(NotImplementedError) as nie:
            self.abstract_function(x=0.0)
        self.assertEqual(str(nie.exception), "Cannot call from base class")

    def test_gradient(self):
        with self.assertRaises(NotImplementedError) as nie:
            self.abstract_function.gradient(x=0.0)
        self.assertEqual(str(nie.exception), "Cannot call from base class")


if __name__ == '__main__':
    unittest.main()
