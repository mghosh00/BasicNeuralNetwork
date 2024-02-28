import unittest
from unittest import TestCase

from neural_network import Neuron


class TestNeuron(TestCase):
    """Tests the `Neuron` class.
    """

    def setUp(self):
        self.neuron = Neuron(3, 4)

    def test_construct(self):
        self.assertEqual((3, 4), self.neuron._id)
        self.assertEqual(0.0, self.neuron._bias)
        self.assertIsNone(self.neuron._value)
        self.assertEqual([], self.neuron.bias_gradients)

    def test_bias_getter_and_setter(self):
        self.assertEqual(0.0, self.neuron.get_bias())
        self.neuron.set_bias(0.4)
        self.assertEqual(0.4, self.neuron._bias)

    def test_value_getter_and_setter(self):
        self.assertIsNone(self.neuron.get_value())
        self.neuron.set_value(0.5)
        self.assertEqual(0.5, self.neuron._value)

    def test_get_id(self):
        self.assertEqual((3, 4), self.neuron.get_id())

    def test_str(self):
        self.assertEqual("Neuron (3, 4)", str(self.neuron))


if __name__ == '__main__':
    unittest.main()
