import unittest
from unittest import TestCase

from neural_network.components.layer import Layer


class TestLayer(TestCase):
    """Tests the `Layer` class
    """

    def setUp(self):
        self.layer = Layer(2, 6)

    def test_construct(self):
        self.assertEqual(2, self.layer._id)
        self.assertEqual(6, self.layer._num_neurons)
        self.assertEqual(6, len(self.layer._neurons))
        for i in range(len(self.layer._neurons)):
            neuron = self.layer._neurons[i]
            self.assertEqual((2, i), neuron.get_id())

    def test_get_id(self):
        self.assertEqual(2, self.layer.get_id())

    def test_get_neurons(self):
        self.assertListEqual(self.layer._neurons, self.layer.get_neurons())

    def test_str(self):
        self.assertEqual("Layer 2", str(self.layer))

    def test_len(self):
        self.assertEqual(6, len(self.layer))


if __name__ == '__main__':
    unittest.main()
