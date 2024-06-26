import unittest
from unittest import TestCase
from unittest import mock

import numpy as np

from neural_network.functions import TransferFunction
from neural_network.functions import ReLU
from neural_network.functions import Softmax
from neural_network.functions import MSELoss

from neural_network.components import Network


class TestNetwork(TestCase):
    """Tests the `Network` class
    """
    pre_activated_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    @mock.patch('random.uniform')
    @mock.patch('random.gauss')
    def setUp(self, mock_gauss, mock_uniform):
        mock_gauss.return_value = 0.2
        mock_uniform.return_value = 0.5
        self.default_network = Network(num_features=3, num_hidden_layers=2,
                                       neuron_counts=[4, 2])
        self.network = Network(num_features=2, num_hidden_layers=3,
                               neuron_counts=[1, 4, 2], regression=False,
                               leak=0.5, learning_rate=0.005, num_classes=3,
                               adaptive=True, gamma=0.8, he_weights=True)
        self.minimal_network = Network(num_features=1, num_hidden_layers=0,
                                       neuron_counts=[], num_classes=2)
        self.regression_network = Network(num_features=2, num_hidden_layers=3,
                                          neuron_counts=[1, 4, 2],
                                          regression=True, leak=0.5,
                                          learning_rate=0.005, num_classes=3,
                                          adaptive=True, gamma=0.8,
                                          he_weights=True)

    def test_construct_erroneous(self):
        with self.assertRaises(ValueError) as ve:
            Network(num_features=1, num_hidden_layers=2,
                    neuron_counts=[3, 4, 5])
        self.assertEqual(str(ve.exception), "neuron_counts (3) must have a "
                                            "length equal to num_hidden_layers"
                                            " (2)")

    def test_construct_defaults(self):
        self.assertEqual(self.default_network._regression, False)
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
        all_neuron_counts = [2, 1, 4, 2, 3]
        for i, layer in enumerate(self.network._layers):
            self.assertEqual(layer.get_id(), i)
            self.assertEqual(len(layer.get_neurons()), all_neuron_counts[i])
        self.assertIn(self.network._input_layer, self.network._layers)
        self.assertIn(self.network._output_layer, self.network._layers)
        for layer in self.network._hidden_layers:
            self.assertIn(layer, self.network._layers)

        # Checking layers for minimal network
        self.assertEqual(len(self.minimal_network._hidden_layers), 0)
        all_neuron_counts = [1, 2, 2]
        for i, layer in enumerate(self.minimal_network._layers):
            self.assertEqual(layer.get_id(), i)
            self.assertEqual(len(layer.get_neurons()), all_neuron_counts[i])
        self.assertIn(self.minimal_network._input_layer,
                      self.minimal_network._layers)
        self.assertIn(self.minimal_network._output_layer,
                      self.minimal_network._layers)

    def test_construct_edges(self):
        # Checking edges
        self.assertEqual(len(self.network._edges), 4)
        for i, left_layer in enumerate(self.network._edges):
            for j, right_neuron in enumerate(left_layer):
                for k, left_neuron in enumerate(right_neuron):
                    edge = self.network._edges[i][j][k]
                    self.assertEqual(edge.get_id(), (i, k, j))
                    # Check for He weights
                    self.assertEqual(edge.get_weight(), 0.2)

        # Check non-He weights for default_network
        for i, left_layer in enumerate(self.default_network._edges):
            for j, right_neuron in enumerate(left_layer):
                for k, left_neuron in enumerate(right_neuron):
                    edge = self.default_network._edges[i][j][k]
                    self.assertEqual(edge.get_id(), (i, k, j))
                    # Check for non-He weights
                    self.assertEqual(edge.get_weight(), 0.5)

    def test_construct_others(self):
        # Checking functions
        self.assertIsInstance(self.network._transfer, TransferFunction)
        self.assertIsInstance(self.network._relu, ReLU)

        # We have different functions depending on whether we use regression
        # or not
        self.assertIsInstance(self.network._softmax, Softmax)
        self.assertFalse(hasattr(self.network, "_mse_loss"))

        # Checking other parameters
        self.assertEqual(self.network._learning_rate, 0.005)
        self.assertEqual(self.network._adaptive, True)
        self.assertEqual(self.network._gamma, 0.8)

    def test_construct_regression_network(self):
        self.assertEqual(self.regression_network._regression, True)

        # Check that num_classes changed to 1 even though 3 was passed to the
        # constructor
        self.assertEqual(len(self.regression_network._output_layer), 1)

        # Check the functions
        self.assertIsInstance(self.regression_network._mse_loss, MSELoss)
        self.assertFalse(hasattr(self.regression_network, "_softmax"))

    def test_forward_pass_one_datapoint_erroneous(self):
        erroneous_data = np.array([1, 2, 3])
        with self.assertRaises(ValueError) as ve:
            self.network.forward_pass_one_datapoint(erroneous_data)
        self.assertEqual(str(ve.exception), "Number of features must match "
                                            "the number of neurons in the "
                                            "input layer (3 != 2)")

    @mock.patch('neural_network.components.network'
                '.Network._calculate_pre_activated_value',
                side_effect=pre_activated_values)
    @mock.patch('neural_network.components.network'
                '.Network._activate_output_layer')
    def test_forward_pass_one_datapoint(self, mock_activate, mock_calculate):
        mock_activate.return_value = [0.2, 0.4, 0.4]
        x = np.array([2, 3])
        softmax_vector = self.network.forward_pass_one_datapoint(x)

        # Check input layer has correct values
        input_layer = self.network._input_layer
        for i in range(len(input_layer)):
            input_neuron = input_layer.get_neurons()[i]
            self.assertEqual(x[i], input_neuron.get_value())

        # All layers but output layer
        main_layers = self.network.get_layers()[:-1]
        # Assert _calculate_pre_activated_value is called
        i = 0
        for left_layer in main_layers[:-1]:
            right_layer = main_layers[left_layer.get_id() + 1]
            for right_neuron in right_layer.get_neurons():
                # Calculates the desired values for each neuron
                mock_calculate.assert_any_call(left_layer, right_neuron)
                self.assertEqual(right_neuron.get_value(),
                                 self.pre_activated_values[i])
                i += 1

        # We have one call for every right_neuron, which will be
        # 1 + 4 + 2 + 3 (hidden layers 1-3 and the output layer)
        self.assertEqual(mock_calculate.call_count, 10)

        # Assert _propagate_softmax_layer is called
        self.assertListEqual([0.2, 0.4, 0.4], softmax_vector)

    @mock.patch('neural_network.components.network'
                '.Network._calculate_pre_activated_value',
                side_effect=pre_activated_values)
    @mock.patch('neural_network.components.network'
                '.Network._activate_output_layer')
    def test_forward_pass_one_datapoint_regression(self, mock_activate,
                                                   mock_calculate):
        mock_activate.return_value = [0.2, 0.4, 0.4]
        mock_calculate.return_value = 0.1
        x = np.array([2, 3])
        pred_list = self.regression_network.forward_pass_one_datapoint(x)

        # Check input layer has correct values
        input_layer = self.regression_network._input_layer
        for i in range(len(input_layer)):
            input_neuron = input_layer.get_neurons()[i]
            self.assertEqual(x[i], input_neuron.get_value())

        # All layers but output layer
        main_layers = self.regression_network.get_layers()[:-1]
        # Assert _calculate_pre_activated_value is called
        i = 0
        for left_layer in main_layers[:-1]:
            right_layer = main_layers[left_layer.get_id() + 1]
            for right_neuron in right_layer.get_neurons():
                # Calculates the desired values for each neuron
                mock_calculate.assert_any_call(left_layer, right_neuron)
                self.assertEqual(right_neuron.get_value(),
                                 self.pre_activated_values[i])
                i += 1

        # We have one call for every right_neuron, which will be
        # 1 + 4 + 2 + 1 (hidden layers 1-3 and the output layer)
        self.assertEqual(mock_calculate.call_count, 8)

        # Check that we get out the final calculated result
        self.assertListEqual(pred_list, [0.8])
        output_neuron = self.regression_network._output_layer.get_neurons()[0]
        self.assertEqual(output_neuron.get_value(), 0.8)

        # Check that mock_activate is never called
        self.assertEqual(mock_activate.call_count, 0)

    def test_calculate_pre_activated_value(self):
        # Here, we wish to control all values involved so that we can
        # track the calculation

        # We choose the left_layer to have 4 neurons, and the right_neuron
        # to be the second one down in its layer
        left_layer = self.network.get_layers()[2]
        left_neurons = left_layer.get_neurons()
        self.assertEqual(len(left_layer), 4)
        right_neuron = self.network.get_layers()[3].get_neurons()[1]

        # This list contains 4 edges all connected to right_neuron from the
        # left_layer
        edges = self.network.get_edges()[2][1]
        self.assertEqual(len(edges), 4)

        # Setting up values
        for i, left_neuron in enumerate(left_neurons):
            # values = 2, 3, 4, 5
            left_neuron.set_value(i + 2)
        for j, edge in enumerate(edges):
            # weights = 2, 1, 0, -1
            edge.set_weight(2 - j)
        right_neuron.set_bias(2)

        # The method will apply the transfer function followed by a Leaky ReLU,
        # with leak = 0.5
        z = self.network._calculate_pre_activated_value(left_layer,
                                                        right_neuron)
        self.assertEqual(4, z)

        # Now try with different weights, so that the transfer is negative
        for j, edge in enumerate(edges):
            # weights = 1, 0, -1, -2
            edge.set_weight(1 - j)
        z = self.network._calculate_pre_activated_value(left_layer,
                                                        right_neuron)
        self.assertEqual(-10, z)

    def test_activate_output_layer(self):
        # Now control all values from the output layer
        z_list = [-2, 0, 2]
        softmax_vector = self.network._activate_output_layer(z_list)

        # Check the normalisation constant is good
        self.assertAlmostEqual(1.15365092,
                               self.network._softmax._normalisation,
                               places=8)

        predicted_vector = [0.01587624, 0.11731043, 0.86681333]
        softmax_neurons = self.network._output_layer.get_neurons()
        for i in range(3):
            self.assertAlmostEqual(softmax_vector[i], predicted_vector[i],
                                   places=8)
            self.assertAlmostEqual(softmax_neurons[i].get_value(),
                                   predicted_vector[i], places=8)

    def test_store_gradient_of_loss_outer_layer(self):
        # index = -1 is outer layer
        # edge location: left_layer = 3, right_neuron = 1, left_neuron = 0
        edge = self.network.get_edges()[-1][1][0]

        # We have 3 classes, so target = 0, 1 or 2 depending on which class
        # the ground truth label corresponds to
        target = 0
        first = True

        # Set the values of the left and right neurons
        edge.get_left_neuron().set_value(0.2)
        right_neuron = edge.get_right_neuron()
        right_neuron.set_value(-0.5)
        edge.loss_gradients = [0.2, 0.1, 0.3]
        right_neuron.bias_gradients = [-0.3, 0.1, 0.4]

        self.network.store_gradient_of_loss(edge, target, first)
        self.assertEqual(edge.get_delta(), -0.5)
        self.assertListEqual(edge.loss_gradients, [0.2, 0.1, 0.3, -0.1])
        self.assertListEqual(right_neuron.bias_gradients, [-0.3, 0.1,
                                                           0.4, -0.5])

        # Now try for a new edge
        edge = self.network.get_edges()[-1][0][1]
        target = 0
        first = False

        # Same inputs as before to provide comparison
        edge.get_left_neuron().set_value(0.2)
        right_neuron = edge.get_right_neuron()
        right_neuron.set_value(-0.5)
        edge.loss_gradients = [0.2, 0.1, 0.3]
        right_neuron.bias_gradients = [-0.3, 0.1, 0.4]

        self.network.store_gradient_of_loss(edge, target, first)
        self.assertEqual(edge.get_delta(), -1.5)
        expected = [0.2, 0.1, 0.3, -0.3]
        for i in range(4):
            self.assertAlmostEqual(edge.loss_gradients[i], expected[i])
        self.assertListEqual(right_neuron.bias_gradients, [-0.3, 0.1, 0.4])

    def test_store_gradient_of_loss_outer_layer_regression(self):
        # index = -1 is outer layer
        # edge location: left_layer = 3, right_neuron = 0, left_neuron = 0
        edge = self.regression_network.get_edges()[-1][0][0]

        # Choose a ground truth value
        target = 8.8
        first = True

        # Set the values of the left and right neurons
        edge.get_left_neuron().set_value(0.2)
        right_neuron = edge.get_right_neuron()
        right_neuron.set_value(8.0)
        edge.loss_gradients = [0.2, 0.1, 0.3]
        right_neuron.bias_gradients = [-0.3, 0.1, 0.4]

        self.regression_network.store_gradient_of_loss(edge, target, first)
        # The delta should be 2 * (o_right - target)
        self.assertEqual(edge.get_delta(), -1.6)
        self.assertListEqual(edge.loss_gradients, [0.2, 0.1, 0.3, -0.32])
        self.assertListEqual(right_neuron.bias_gradients, [-0.3, 0.1,
                                                           0.4, -1.6])

        # Now try for a new edge
        edge = self.regression_network.get_edges()[-1][0][1]
        target = 8.8
        first = False

        # Same inputs as before to provide comparison
        edge.get_left_neuron().set_value(0.2)
        right_neuron = edge.get_right_neuron()
        right_neuron.set_value(8.0)
        edge.loss_gradients = [0.2, 0.1, 0.3]
        right_neuron.bias_gradients = [-0.3, 0.1, 0.4]

        self.regression_network.store_gradient_of_loss(edge, target, first)
        self.assertEqual(edge.get_delta(), -1.6)
        expected = [0.2, 0.1, 0.3, -0.32]
        for i in range(4):
            self.assertAlmostEqual(edge.loss_gradients[i], expected[i])
        self.assertListEqual(right_neuron.bias_gradients, [-0.3, 0.1, 0.4])

    def test_store_gradient_of_loss_hidden_layer(self):
        # edge location: left_layer = 2, right_neuron = 1, left_neuron = 0
        edge = self.network.get_edges()[2][1][0]

        # Set the values of the left and right neurons
        edge.get_left_neuron().set_value(0.2)
        right_neuron = edge.get_right_neuron()
        right_neuron.set_value(-0.5)
        edge.loss_gradients = [0.2, 0.1, 0.3]
        right_neuron.bias_gradients = [-0.3, 0.1, 0.4]

        # There are 3 edges connecting the right_neuron to the next layer
        # along, and we must take these values into account also
        new_edges = [self.network.get_edges()[3][j][1] for j in range(3)]
        for j, new_edge in enumerate(new_edges):
            new_edge.set_weight(j + 1)
            new_edge.set_delta(-j)

        target = 0
        first = True
        self.network.store_gradient_of_loss(edge, target, first)

        # 1 * 0 + 2 * (-1) + 3 * (-2)
        factor = -8
        relu_grad = 0.5
        self.assertEqual(edge.get_delta(), factor * relu_grad)
        self.assertListEqual(edge.loss_gradients, [0.2, 0.1, 0.3, -0.8])
        self.assertListEqual(right_neuron.bias_gradients, [-0.3, 0.1,
                                                           0.4, -4])

    def test_back_propagate_weight_no_momentum(self):
        # The default_network has _adaptive = False
        edge = self.default_network.get_edges()[1][1][3]

        # Current weight
        edge.set_weight(1)

        # The loss gradients picked up by the batch
        edge.loss_gradients = [-0.1, 0.0, 0.1, 0.2, 0.3]
        self.default_network.back_propagate_weight(edge)

        # learning_rate is 0.01
        self.assertListEqual(edge.loss_gradients, [])
        self.assertEqual(edge.get_weight(), 1 - 0.01 * 0.1)

    def test_back_propagate_weight_with_momentum(self):
        # The network has _adaptive = True
        edge = self.network.get_edges()[1][1][0]

        # Current weight and velocity for edge
        edge.set_weight(1)
        edge.set_velocity(-2)

        # The loss gradients picked up by the batch
        edge.loss_gradients = [-0.1, 0.0, 0.1, 0.2, 0.3]
        self.network.back_propagate_weight(edge)

        # learning_rate is 0.005 and gamma is 0.8
        self.assertListEqual(edge.loss_gradients, [])
        self.assertEqual(edge.get_weight(), 1 + 2 * 0.8 - 0.005 * 0.1)
        self.assertEqual(edge.get_velocity(), 0.005 * 0.1 - 2 * 0.8)

    def test_back_propagate_bias(self):
        neuron = self.network.get_layers()[3].get_neurons()[0]

        # Current bias for neuron
        neuron.set_bias(2)

        # The bias gradients picked up by the batch
        neuron.bias_gradients = [0.1, 0.2, 0.3, 0.4]
        self.network.back_propagate_bias(neuron)

        # Check they have been reset
        self.assertListEqual(neuron.bias_gradients, [])

        # Recall learning_rate = 0.005
        self.assertEqual(neuron.get_bias(), 2 - 0.005 * 0.25)

    @mock.patch('matplotlib.pyplot.clf')
    @mock.patch('matplotlib.pyplot.savefig')
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('matplotlib.pyplot.title')
    @mock.patch('networkx.multipartite_layout')
    @mock.patch('networkx.draw_networkx')
    @mock.patch('networkx.Graph')
    def test_visualise_network_default(self, mock_graph_init, mock_draw,
                                       mock_layout, mock_title, mock_show,
                                       mock_save, mock_clf):
        tuple_edges = [("0,0", "1,0"), ("0,1", "1,0"), ("0,2", "1,0"),
                       ("0,0", "1,1"), ("0,1", "1,1"), ("0,2", "1,1"),
                       ("0,0", "1,2"), ("0,1", "1,2"), ("0,2", "1,2"),
                       ("0,0", "1,3"), ("0,1", "1,3"), ("0,2", "1,3"),
                       ("1,0", "2,0"), ("1,1", "2,0"), ("1,2", "2,0"),
                       ("1,3", "2,0"), ("1,0", "2,1"), ("1,1", "2,1"),
                       ("1,2", "2,1"), ("1,3", "2,1"), ("2,0", "3,0"),
                       ("2,1", "3,0"), ("2,0", "3,1"), ("2,1", "3,1")]
        mock_layout.return_value = {}
        self.default_network.visualise_network()
        mock_graph = mock_graph_init()
        for edge in tuple_edges:
            mock_graph.add_edge.assert_any_call(edge[0], edge[1])
            mock_graph.add_node.assert_any_call(edge[0], layer=int(edge[0][0]))
            mock_graph.add_node.assert_any_call(edge[1], layer=int(edge[1][0]))
        self.assertEqual(mock_graph.add_edge.call_count, 24)
        self.assertEqual(mock_graph.add_node.call_count, 48)
        mock_draw.assert_called_once_with(mock_graph, pos={}, node_size=480,
                                          node_color='green')
        mock_title.assert_called_once_with("Network")
        mock_show.assert_called_once()
        mock_save.assert_called_once_with("network.png")
        mock_clf.assert_called_once()

    @mock.patch('matplotlib.pyplot.clf')
    @mock.patch('matplotlib.pyplot.savefig')
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('matplotlib.pyplot.title')
    @mock.patch('networkx.multipartite_layout')
    @mock.patch('networkx.draw_networkx')
    @mock.patch('networkx.Graph')
    def test_visualise_network(self, mock_graph_init, mock_draw,
                               mock_layout, mock_title, mock_show, mock_save,
                               mock_clf):
        tuple_edges = [("0,0", "1,0"), ("0,1", "1,0"), ("0,2", "1,0"),
                       ("0,0", "1,1"), ("0,1", "1,1"), ("0,2", "1,1"),
                       ("0,0", "1,2"), ("0,1", "1,2"), ("0,2", "1,2"),
                       ("0,0", "1,3"), ("0,1", "1,3"), ("0,2", "1,3"),
                       ("1,0", "2,0"), ("1,1", "2,0"), ("1,2", "2,0"),
                       ("1,3", "2,0"), ("1,0", "2,1"), ("1,1", "2,1"),
                       ("1,2", "2,1"), ("1,3", "2,1"), ("2,0", "3,0"),
                       ("2,1", "3,0"), ("2,0", "3,1"), ("2,1", "3,1")]
        mock_layout.return_value = {}
        self.default_network.visualise_network(title="test")
        mock_graph = mock_graph_init()
        for edge in tuple_edges:
            mock_graph.add_edge.assert_any_call(edge[0], edge[1])
            mock_graph.add_node.assert_any_call(edge[0], layer=int(edge[0][0]))
            mock_graph.add_node.assert_any_call(edge[1], layer=int(edge[1][0]))
        self.assertEqual(mock_graph.add_edge.call_count, 24)
        self.assertEqual(mock_graph.add_node.call_count, 48)
        mock_draw.assert_called_once_with(mock_graph, pos={}, node_size=480,
                                          node_color='green')
        mock_title.assert_called_once_with("Network: test")
        mock_show.assert_called_once()
        mock_save.assert_called_once_with("network_test.png")
        mock_clf.assert_called_once()

    def test_get_edges(self):
        self.assertListEqual(self.network._edges,
                             self.network.get_edges())

    def test_get_main_layers(self):
        self.assertListEqual(self.network._layers,
                             self.network.get_layers())

    def test_get_neuron_counts(self):
        self.assertListEqual([2, 1, 4, 2, 3],
                             self.network.get_neuron_counts())

    def test_do_regression(self):
        self.assertTrue(self.regression_network.is_regressor())
        self.assertFalse(self.network.is_regressor())


if __name__ == '__main__':
    unittest.main()
