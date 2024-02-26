from typing import List
import numpy as np

from neural_network.functions import TransferFunction
from neural_network.functions import ReLU
from neural_network.functions import Softmax
from neural_network.functions import Loss

from .neuron import Neuron
from .edge import Edge
from .layer import Layer


class Network:
    """Class to represent the whole `Network`
    """

    def __init__(self, num_features: int, num_hidden_layers: int,
                 neuron_counts: List[int], leak: float = 0.01,
                 learning_rate: float = 0.01, num_classes: int = 2,
                 adaptive: bool = False, gamma: float = 0.9):
        """Constructor method

        Parameters
        ----------
        num_features : int
            The number of features for the network
        num_hidden_layers : int
            The total number of hidden `Layers` in the `Network`
        neuron_counts : List[int]
            A list of numbers of `Neurons` for each hidden `Layer`
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
        """
        if not num_hidden_layers == len(neuron_counts):
            raise ValueError("neuron_counts must have a length equal to"
                             "num_hidden_layers")
        self._num_features = num_features
        self._num_hidden_layers = num_hidden_layers
        self._neuron_counts = neuron_counts
        self._input_layer = Layer(0, num_features)
        self._hidden_layers = [Layer(i, neuron_counts[i - 1])
                               for i in range(1, num_hidden_layers + 1)]
        self._output_layer = Layer(num_hidden_layers + 1, num_classes)
        self._softmax_layer = Layer(num_hidden_layers + 2, num_classes)
        self._main_layers = ([self._input_layer] + self._hidden_layers
                             + [self._output_layer])
        self._edges = []
        for i, left_layer in enumerate(self._main_layers[:-1]):
            right_layer = self._main_layers[i + 1]
            layer_list = []
            for right_neuron in right_layer.get_neurons():
                edge_list = [Edge(left_neuron, right_neuron)
                             for left_neuron in left_layer.get_neurons()]
                layer_list.append(edge_list)
            self._edges.append(layer_list)
        self._softmax_edges = [Edge(self._output_layer.get_neurons()[i],
                                    self._softmax_layer.get_neurons()[i])
                               for i in range(num_classes)]
        self._transfer = TransferFunction()
        self._relu = ReLU(leak)
        self._softmax = Softmax()
        self._loss = Loss()
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
            The softmax probabilities of each class
        """
        input_neurons = self._input_layer.get_neurons()
        if not len(x) == len(input_neurons):
            raise ValueError(f"Number of features must match the number of "
                             f"neurons in the input layer ({len(x)} != "
                             f"{len(input_neurons)})")
        for j, neuron in enumerate(input_neurons):
            neuron.set_value(x[j])
        for i, left_layer in enumerate(self._main_layers[:-1]):
            right_layer = self._main_layers[i + 1]
            for k, right_neuron in enumerate(right_layer.get_neurons()):
                left_neurons = left_layer.get_neurons()
                edges = self._edges[i][k]
                o_list = [neuron.get_value() for neuron in left_neurons]
                w_list = [edge.get_weight() for edge in edges]
                bias = right_neuron.get_bias()
                a = self._transfer(o_list, w_list + [bias])
                right_neuron.set_value(self._relu(a))

        values = [neuron.get_value()
                  for neuron in self._output_layer.get_neurons()]
        self._softmax.normalisation(values)
        softmax_neurons = self._softmax_layer.get_neurons()
        softmax_vector = []
        for j, value in enumerate(values):
            softmax = self._softmax(value)
            softmax_vector.append(softmax)
            softmax_neurons[j].set_value(softmax)

        return softmax_vector

    def store_gradient_of_loss(self, edge: Edge, target: int, first: bool):
        """Calculates the gradient of the loss function with respect to one
        weight (assigned to the edge) based on the values at edges of future
        layers. One part of the back propagation process

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
        o_left = edge.get_left_neuron().get_value()
        o_right = right_neuron.get_value()
        relu_grad = self._relu.gradient(o_right)
        right_index, row = right_neuron.get_id()
        if left_layer_index == self._num_hidden_layers + 1:
            edge.loss_gradients.append(o_right - int(row == target))
        elif left_layer_index == self._num_hidden_layers:
            delta = self._softmax_edges[row].loss_gradients[-1] * relu_grad
            edge.loss_gradients.append(o_left * delta)
            if first:
                right_neuron.bias_gradients.append(delta)
        else:
            next_layer = self._main_layers[right_index + 1]
            next_edges = [self._edges[right_index][j][row]
                          for j in range(len(next_layer.get_neurons()))]
            factor = sum([new_edge.get_weight() * new_edge.loss_gradients[-1]
                          for new_edge in next_edges])
            edge.loss_gradients.append(o_left * factor * relu_grad)
            if first:
                right_neuron.bias_gradients.append(factor * relu_grad)

    def back_propagate(self, edge: Edge):
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

        node : Node
            The `Node` whose bias we are interested in updating
        """
        current_bias = neuron.get_bias()
        batch_size = len(neuron.bias_gradients)
        avg_bias_gradient = sum(neuron.bias_gradients) / batch_size
        neuron.set_bias(current_bias - self._learning_rate
                        * avg_bias_gradient)
        neuron.bias_gradients = []

    def get_edges(self) -> List[List[List[Edge]]]:
        """Getter method for edges

        Returns
        -------
        List[List[List[Edge]]]
            A list of `Edges`
        """
        return self._edges

    def get_main_layers(self) -> List[Layer]:
        """Getter method for main layers

        Returns
        -------
        List[Layer]
            A list of main `Layers`
        """
        return self._main_layers

    def get_softmax_edges(self) -> List[Edge]:
        """Getter method for softmax edges

        Returns
        -------
        typing.List
            A list of softmax `Edges`
        """
        return self._softmax_edges

    def get_neuron_counts(self) -> List[int]:
        """Getter method for neuron_counts

        Returns
        -------
        List[int]
            A list of numbers of neurons per layer
        """
        return ([self._num_features] + self._neuron_counts
                + [self._num_features])
