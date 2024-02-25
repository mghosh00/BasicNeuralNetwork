import math
import typing
import pandas as pd

from neural_network.functions import Partitioner
from neural_network.functions import Loss
from neural_network.components import Network

from .plotter import Plotter

class Trainer:
    """Class to train a neural network
    """

    def __init__(self, network: Network, data: pd.DataFrame, num_epochs: int,
                 batch_size: int, classification: bool = True):
        """Constructor method

        Parameters
        ----------
        network : Network
            The neural network to train
        data : pd.DataFrame
            All the training data for the `Network`
        num_epochs : int
            The number of epochs we are training for
        batch_size : int
            The number of datapoints used in each epoch
        classification : bool
            If `True` then we are classifying, otherwise it will be regression
        """
        self._network = network

        # Ensure that number of input nodes equals number of features
        n = len(data.columns)
        if network.get_neuron_counts()[0] != n:
            raise ValueError("Number of features must match number of initial "
                             "neurons")

        # Renaming of columns
        data.columns = [f'x_{i}' for i in range(n - 1)] + ['y']
        data['y_hat'] = [0] * len(data)
        self._data = data
        self._num_epochs = num_epochs

        # Ensure that batch_size is not too big
        if batch_size > len(data):
            raise ValueError("Batch size must be smaller than number of "
                             "datapoints")
        self._batch_size = batch_size
        self._classification = classification
        self._loss = Loss()
        self._partitioner = Partitioner(len(data), batch_size)
        self._plotter = Plotter()

    def forward_pass_one_batch(self, batch_ids: typing.List) -> float:
        """Performs the forward pass for one batch of the data. Also finds the
        gradient of the loss for all points and stores them in preparation of
        back propagation.

        Parameters
        ----------
        batch_ids : typing.List
            The random list of ids for the current batch

        Returns
        -------
        float
            The total loss of the batch (to keep track)
        """
        total_loss = 0
        for i in batch_ids:
            labelled_point = self._data.loc[i].to_numpy()[1:]
            x, y = labelled_point[:-2], int(labelled_point[-2])
            # Do the forward pass and save the predicted value to the df
            y_hat = self._network.forward_pass_one_datapoint(x)
            total_loss += self._loss(y_hat, y)
            self._data.at[i, 'y_hat'] = max(range(len(y_hat)),
                                            key=y_hat.__getitem__)

            # Take gradients of loss and store them in the edges (backwards)
            for softmax_edge in self._network.get_softmax_edges():
                self._network.store_gradient_of_loss(softmax_edge, y, False)

            edges = self._network.get_edges()
            for edge_layer in reversed(edges):
                for right_node in edge_layer:
                    first = True
                    for edge in right_node:
                        self._network.store_gradient_of_loss(edge, y, first)
                        first = False

        # Return the total loss for this batch
        return total_loss

    def back_propagate_one_batch(self):
        """Performs back propagation for one batch of datapoints (stored within
        the memory of the edges).
        """
        edges = self._network.get_edges()
        for layer in reversed(edges):
            for right_node in layer:
                for edge in right_node:
                    self._network.back_propagate(edge)

        layers = self._network.get_main_layers()
        for layer in layers[1:]:
            for neuron in layer.get_neurons():
                self._network.back_propagate_bias(neuron)

    def train(self):
        """Performs training of the network
        """
        factor = self._num_epochs / 100
        for epoch in range(self._num_epochs):
            total_loss = 0
            batch_partition = self._partitioner()
            for iteration in range(math.ceil(len(self._data) /
                                             self._batch_size)):
                batch_ids = batch_partition[iteration]
                total_loss += self.forward_pass_one_batch(batch_ids)
                self.back_propagate_one_batch()
            if epoch % factor == 0:
                print(f"Epoch: {epoch}")
                print(f"Loss: {total_loss / len(self._data)}")
                pd.set_option('display.max_rows', None)
                # print(self._data)

    def generate_plot(self):
        self._plotter.plot_predictions(self._data)

    def generate_gif(self):
        self._plotter.plot_predictions_gif()
