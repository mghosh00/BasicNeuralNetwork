import math
import pandas as pd

from neural_network.components import Network

from .plotter import Plotter
from .abstract_simulator import AbstractSimulator
from .validator import Validator


class Trainer(AbstractSimulator):
    """Class to train a neural network
    """

    def __init__(self, network: Network, data: pd.DataFrame, num_epochs: int,
                 batch_size: int, validator: Validator = None,
                 weighted: bool = False, classification: bool = True):
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
        validator : Validator
            The validator used (if any)
        weighted : bool
            If `True` then we use the WeightedPartitioner, otherwise we use
            the standard Partitioner
        classification : bool
            If `True` then we are classifying, otherwise it will be regression
        """
        super().__init__(network, data, batch_size, weighted, classification)
        self._num_epochs = num_epochs
        self._validator = validator
        self._plotter = Plotter()

        columns = ['Training']
        if self._validator:
            columns.append('Validation')
        self._loss_df = pd.DataFrame(columns=columns)

    def store_gradients(self, _id: int):
        """Stores the gradients of the loss functions after a forward pass

        Parameters
        ----------
        _id : int
            The id of the datapoint

        Returns
        -------
        float
            The total loss of the batch (to keep track)
        """
        y = int(self._data.loc[_id, 'y'])

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

    def run(self):
        """Performs training of the network
        """
        factor = int(self._num_epochs / 100)
        for epoch in range(self._num_epochs):
            total_loss = 0
            batch_partition = self._partitioner()
            for iteration in range(math.ceil(len(self._data) /
                                             self._batch_size)):
                batch_ids = batch_partition[iteration]
                total_loss += self.forward_pass_one_batch(batch_ids)
                self.back_propagate_one_batch()
            loss = total_loss / len(self._data)
            if epoch % factor == 0:
                print(f"Epoch: {epoch}")
                print(f"Loss: {loss}")

            self._loss_df.at[epoch, 'Training'] = loss
            if self._validator:
                validation_loss = self._validator.validate(factor)
                self._loss_df.at[epoch, 'Validation'] = validation_loss
                # pd.set_option('display.max_rows', None)
                # print(self._data)

    def generate_scatter(self, title: str = ''):
        """Creates scatter plot from the data and their predicted values

        Parameters
        ----------
        title : str
            An optional title to append to the plot
        """
        super().generate_scatter(f'training_{title}')

    def generate_loss_plot(self, title: str = ''):
        Plotter.plot_loss(self._loss_df, title)

    def generate_gif(self):
        self._plotter.plot_predictions_gif()
