from typing import List
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from neural_network.functions import TransferFunction
from neural_network.functions import ReLU
from neural_network.functions import Softmax
from neural_network.functions import MSELoss

from .neuron import Neuron
from .edge import Edge
from .layer import Layer


class Network:
    """Class to represent the whole `Network`
    """

    def __init__(self, num_features: int, num_hidden_layers: int,
                 neuron_counts: List[int], regression: bool = False,
                 leak: float = 0.01, learning_rate: float = 0.01,
                 num_classes: int = 2, adaptive: bool = False,
                 gamma: float = 0.9, he_weights: bool = False):
        """Constructor method

        Parameters
        ----------
        num_features : int
            The number of features for the network
        num_hidden_layers : int
            The total number of hidden `Layers` in the `Network`
        neuron_counts : List[int]
            A list of numbers of `Neurons` for each hidden `Layer`
        regression : bool
            Whether we are performing regression or not (if `False` we are
            performing classification)
        leak : float
            The leak rate of LeakyReLU
        learning_rate : float
            The learning rate of the network
        num_classes : int
            The number of classes for the classification task
        adaptive : bool
            Whether we wish to have an adaptive learning rate or not
        gamma : float
            The adaptive learning rate parameter
        he_weights : bool
            Whether we wish to initialise the weights according to He or not
        """
        if not num_hidden_layers == len(neuron_counts):
            raise ValueError(f"neuron_counts ({len(neuron_counts)}) must have "
                             f"a length equal to num_hidden_layers "
                             f"({num_hidden_layers})")
        self._num_features = num_features
        self._num_hidden_layers = num_hidden_layers
        self._neuron_counts = neuron_counts
        self._regression = regression
        # If we are doing regression set num_classes to 1, no matter what the
        # input was
        if regression:
            num_classes = 1

        # Layers
        self._input_layer = Layer(0, num_features)
        self._hidden_layers = [Layer(i, neuron_counts[i - 1])
                               for i in range(1, num_hidden_layers + 1)]
        self._output_layer = Layer(num_hidden_layers + 1, num_classes)
        self._layers = ([self._input_layer] + self._hidden_layers
                        + [self._output_layer])

        # Edges
        self._edges = []
        for i, left_layer in enumerate(self._layers[:-1]):
            right_layer = self._layers[i + 1]
            layer_list = []
            for right_neuron in right_layer.get_neurons():
                edge_list = [Edge(left_neuron, right_neuron)
                             for left_neuron in left_layer.get_neurons()]
                layer_list.append(edge_list)
                if he_weights:
                    n = len(left_layer)
                    for edge in edge_list:
                        edge.set_weight(random.gauss(0.0,
                                                     math.sqrt(2 / n)))
            self._edges.append(layer_list)

        # Functions and other parameters
        self._transfer = TransferFunction()
        self._relu = ReLU(leak)
        if regression:
            self._mse_loss = MSELoss()
        else:
            self._softmax = Softmax()
        self._learning_rate = learning_rate
        self._adaptive = adaptive
        self._gamma = gamma

    def forward_pass_one_datapoint(self, x: np.array) -> List[float]:
        """Performs a forward pass for one datapoint, excluding the ground
        truth value. This method returns the predicted value in the final
        neuron in the output layer.

        Parameters
        ----------
        x : np.array
            The input value, with all features

        Returns
        -------
        List[float]
            The softmax probabilities of each class (for classification) or
            the predicted regression value (for regression)
        """
        if not len(x) == len(self._input_layer):
            raise ValueError(f"Number of features must match the number of "
                             f"neurons in the input layer ({len(x)} != "
                             f"{len(self._input_layer)})")

        # Input layer
        input_neurons = self._input_layer.get_neurons()
        for j, neuron in enumerate(input_neurons):
            neuron.set_value(x[j])

        # This is for the output layer
        z_output_layer = []
        # Hidden layers and output layer
        for left_layer in self._layers[:-1]:
            right_layer = self._layers[left_layer.get_id() + 1]

            if right_layer.get_id() != self._output_layer.get_id():
                for right_neuron in right_layer.get_neurons():
                    # Calculates the desired values for each neuron
                    z = self._calculate_pre_activated_value(left_layer,
                                                            right_neuron)
                    # Uses ReLU activation for the neuron
                    right_neuron.set_value(self._relu(z))
            else:
                if self._regression:
                    # We only have one output neuron with linear activation
                    # for a regression network
                    output_neuron = self._output_layer.get_neurons()[0]
                    z = self._calculate_pre_activated_value(left_layer,
                                                            output_neuron)
                    output_neuron.set_value(z)
                    # Here we exit the method, returning the value of the
                    # output neuron
                    return [z]
                else:
                    for right_neuron in right_layer.get_neurons():
                        # Calculate the desired values for each neuron
                        z = self._calculate_pre_activated_value(left_layer,
                                                                right_neuron)
                        z_output_layer.append(z)

        # Output layer
        # Activates the output layer using Softmax activation
        softmax_vector = self._activate_output_layer(z_output_layer)
        return softmax_vector

    def _calculate_pre_activated_value(self, left_layer: Layer,
                                       right_neuron: Neuron) -> float:
        """Given a `left_layer` and a `right_neuron`, this calculates the
        activation function and value from the `left_layer` and propagates
        this value to the `right_neuron`

        Parameters
        ----------
        left_layer : Layer
            The current `left_layer` in forward propagation
        right_neuron : Neuron
            The current `right_neuron` in forward propagation
        """
        left_neurons = left_layer.get_neurons()
        i, j = right_neuron.get_id()

        # All edges connecting the left_layer to the right_neuron
        edges = self._edges[i - 1][j]

        # Lists of values and weights, with the bias
        o_list = [neuron.get_value() for neuron in left_neurons]
        w_list = [edge.get_weight() for edge in edges]
        bias = right_neuron.get_bias()

        # Return the pre-activated value for this neuron
        return self._transfer(o_list, w_list + [bias])

    def _activate_output_layer(self, z_list: List[float]) -> List[float]:
        """Activates the values from the `output_layer` using the Softmax
        activation function.

        Returns
        -------
        List[float]
            The `List` of softmax probabilities
        """
        self._softmax.normalise(z_list)
        softmax_neurons = self._output_layer.get_neurons()
        softmax_vector = []
        for j, value in enumerate(z_list):
            softmax = self._softmax(value)
            softmax_vector.append(softmax)
            softmax_neurons[j].set_value(softmax)

        return softmax_vector

    def store_gradient_of_loss(self, edge: Edge, target: float, first: bool):
        """Calculates the gradient of the loss function with respect to one
        weight (assigned to the edge) based on the values at edges of future
        layers. One part of the back propagation process.

        edge : Edge
            The `Edge` containing the weight we are interested in
        target : int
            The target value for the final output node for this specific
            datapoint
        first : bool
            Determines whether we find the bias gradient or not
        """
        left_layer_index = edge.get_id()[0]
        right_neuron = edge.get_right_neuron()

        # Value of left neuron and right neuron
        o_left = edge.get_left_neuron().get_value()
        o_right = right_neuron.get_value()

        right_index, row = right_neuron.get_id()

        # Output layer
        if left_layer_index == self._num_hidden_layers:
            if self._regression:
                delta = self._mse_loss.gradient(o_right, target)
            else:
                delta = o_right - int(row == int(target))
            edge.set_delta(delta)
            edge.loss_gradients.append(round(o_left * delta, 8))
            if first:
                right_neuron.bias_gradients.append(delta)

        # Hidden layers
        else:
            next_layer = self._layers[right_index + 1]
            next_edges = [self._edges[right_index][j][row]
                          for j in range(len(next_layer))]
            factor = sum([new_edge.get_weight() * new_edge.get_delta()
                          for new_edge in next_edges])

            # Constant (either +1 or self._leak)
            relu_grad = self._relu.gradient(o_right)
            delta = factor * relu_grad
            edge.set_delta(delta)
            edge.loss_gradients.append(o_left * delta)
            if first:
                right_neuron.bias_gradients.append(delta)

    def back_propagate_weight(self, edge: Edge):
        """Uses the loss gradients of all datapoints (for this specific edge)
        to perform gradient descent and calculate a new weight for this edge.

        edge : Edge
            The `Edge` whose weight we are interested in updating
        """
        current_weight = edge.get_weight()
        batch_size = len(edge.loss_gradients)
        avg_loss_gradient = sum(edge.loss_gradients) / batch_size
        if self._adaptive:
            velocity = (self._gamma * edge.get_velocity()
                        + self._learning_rate * avg_loss_gradient)
            edge.set_weight(current_weight - velocity)
            edge.set_velocity(velocity)
        else:
            edge.set_weight(current_weight - self._learning_rate
                            * avg_loss_gradient)
        edge.loss_gradients = []

    def back_propagate_bias(self, neuron: Neuron):
        """Uses the bias gradients of all datapoints (for this specific neuron)
        to perform gradient descent and calculate a new bias for this neuron.

        neuron : Neuron
            The `Neuron` whose bias we are interested in updating
        """
        current_bias = neuron.get_bias()
        batch_size = len(neuron.bias_gradients)
        avg_bias_gradient = sum(neuron.bias_gradients) / batch_size
        neuron.set_bias(current_bias - self._learning_rate
                        * avg_bias_gradient)
        neuron.bias_gradients = []

    def visualise_network(self, title: str = ''):
        """Uses the networkx package to visualise the network.

        Parameters
        ----------
        title : str [Optional, Default='']
            An optional title for the network.
        """
        graph = nx.Graph()
        for edge_layer in self._edges:
            for right_neuron in edge_layer:
                for edge in right_neuron:
                    left_id = edge.get_left_neuron().get_id()
                    right_id = edge.get_right_neuron().get_id()
                    left_id_str = f"{left_id[0]},{left_id[1]}"
                    right_id_str = f"{right_id[0]},{right_id[1]}"
                    graph.add_node(left_id_str, layer=left_id[0])
                    graph.add_node(right_id_str, layer=right_id[0])
                    graph.add_edge(left_id_str, right_id_str)
        pos = nx.multipartite_layout(graph, subset_key="layer")
        nx.draw_networkx(graph, pos=pos,
                         node_size=480, node_color='green')
        joining_str = f"_{title}" if title else ""
        title_str = f": {title}" if title else ""
        plt.title(f"Network{title_str}")
        plt.savefig(f"network{joining_str}.png")
        plt.clf()

    def get_edges(self) -> List[List[List[Edge]]]:
        """Getter method for edges

        Returns
        -------
        List[List[List[Edge]]]
            A list of `Edges`
        """
        return self._edges

    def get_layers(self) -> List[Layer]:
        """Getter method for layers

        Returns
        -------
        List[Layer]
            A list of `Layers`
        """
        return self._layers

    def get_neuron_counts(self) -> List[int]:
        """Getter method for neuron_counts.

        Returns
        -------
        List[int]
            A list of numbers of neurons per layer
        """
        return ([self._num_features] + self._neuron_counts
                + [len(self._output_layer)])

    def is_regressor(self) -> bool:
        """Getter method for regression.

        Returns
        -------
        bool
            Whether we do regression or classification
        """
        return self._regression
