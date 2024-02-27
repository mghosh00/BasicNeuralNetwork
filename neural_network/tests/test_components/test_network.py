import unittest
from unittest import TestCase

from neural_network import TransferFunction
from neural_network import ReLU
from neural_network import Softmax
from neural_network import Loss

from neural_network import Neuron
from neural_network import Edge
from neural_network import Layer
from neural_network import Network


class TestNetwork(TestCase):
    """Tests the `Network` class
    """

    def setUp(self):
        self.default_network = Network(num_features=3, num_hidden_layers=2,
                                       neuron_counts=[4, 2])
        self.network = Network(num_features=2, num_hidden_layers=3,
                               neuron_counts=[1, 4, 2], leak=0.5,
                               learning_rate=0.005, num_classes=3,
                               adaptive=True, gamma=0.8)
        self.minimal_network = Network(num_features=1, num_hidden_layers=0,
                                       neuron_counts=[], num_classes=2)

    def test_construct_erroneous(self):
        with self.assertRaises(ValueError) as ve:
            Network(num_features=1, num_hidden_layers=2,
                    neuron_counts=[3, 4, 5])
        self.assertEqual(str(ve.exception), "neuron_counts (3) must have a "
                                            "length equal to num_hidden_layers"
                                            " (2)")

    def test_construct_defaults(self):
        self.assertEqual(self.default_network._relu._leak, 0.01)
        self.assertEqual(self.default_network._learning_rate, 0.01)

        # Tests that num_classes is 2
        self.assertEqual(len(self.default_network._output_layer), 2)
        self.assertEqual(self.default_network._adaptive, False)
        self.assertEqual(self.default_network._gamma, 0.9)

    def test_construct_layers(self):
        self.assertEqual(self.network._num_features, 2)
        self.assertEqual(self.network._num_hidden_layers, 3)
        self.assertListEqual(self.network._neuron_counts, [1, 4, 2])

        # Checking layers for main network
        self.assertEqual(len(self.network._hidden_layers), 3)
        all_neuron_counts = [2, 1, 4, 2, 3, 3]
        for i, layer in enumerate(self.network._main_layers
                                  + [self.network._softmax_layer]):
            self.assertEqual(layer.get_id(), i)
            self.assertEqual(len(layer.get_neurons()), all_neuron_counts[i])
        self.assertIn(self.network._input_layer, self.network._main_layers)
        self.assertIn(self.network._output_layer, self.network._main_layers)
        for layer in self.network._hidden_layers:
            self.assertIn(layer, self.network._main_layers)

        # Checking layers for minimal network
        self.assertEqual(len(self.minimal_network._hidden_layers), 0)
        all_neuron_counts = [1, 2, 2]
        for i, layer in enumerate(self.minimal_network._main_layers
                                  + [self.minimal_network._softmax_layer]):
            self.assertEqual(layer.get_id(), i)
            self.assertEqual(len(layer.get_neurons()), all_neuron_counts[i])
        self.assertIn(self.minimal_network._input_layer,
                      self.minimal_network._main_layers)
        self.assertIn(self.minimal_network._output_layer,
                      self.minimal_network._main_layers)

    def test_construct_edges(self):
        # Checking edges
        self.assertEqual(len(self.network._edges), 4)
        for i, left_layer in enumerate(self.network._edges):
            for j, right_neuron in enumerate(left_layer):
                for k, left_neuron in enumerate(right_neuron):
                    edge = self.network._edges[i][j][k]
                    self.assertEqual(edge.get_id(), (i, k, j))

        self.assertEqual(len(self.network._softmax_edges), 3)
        for i, edge in enumerate(self.network._softmax_edges):
            self.assertEqual(edge.get_id(), (4, i, i))

    def test_construct_others(self):
        # Checking functions
        self.assertIsInstance(self.network._transfer, TransferFunction)
        self.assertIsInstance(self.network._relu, ReLU)
        self.assertIsInstance(self.network._softmax, Softmax)
        self.assertIsInstance(self.network._loss, Loss)

        # Checking other parameters
        self.assertEqual(self.network._learning_rate, 0.005)
        self.assertEqual(self.network._adaptive, True)
        self.assertEqual(self.network._gamma, 0.8)


if __name__ == '__main__':
    unittest.main()
