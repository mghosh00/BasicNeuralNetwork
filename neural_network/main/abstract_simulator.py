from typing import List
import pandas as pd

from neural_network.util import Partitioner
from neural_network.util import WeightedPartitioner
from neural_network.functions import Loss
from neural_network.components import Network

from .plotter import Plotter


class AbstractSimulator:
    """Base class for trainer, tester and validator
    """

    def __init__(self, network: Network, data: pd.DataFrame, batch_size: int,
                 weighted: bool = False, classification: bool = True):
        """Constructor method

        Parameters
        ----------
        network : Network
            The neural network to train
        data : pd.DataFrame
            All the data for the `Network`
        batch_size : int
            The number of datapoints used in each epoch
        weighted : bool
            If `True` then we use the WeightedPartitioner, otherwise we use
            the standard Partitioner
        classification : bool
            If `True` then we are classifying, otherwise it will be regression
        """
        self._network = network

        # Ensure that number of input nodes equals number of features
        n = len(data.columns) - 1
        m = network.get_neuron_counts()[0]
        if m != n:
            raise ValueError(f"Number of features must match number of "
                             f"initial neurons (features = {n}, initial "
                             f"neurons = {m})")

        # Renaming of columns
        data.columns = [f'x_{i + 1}' for i in range(n)] + ['y']
        data['y_hat'] = [0] * len(data)
        self._data = data

        # Ensure that batch_size is not too big
        if batch_size > len(data):
            raise ValueError("Batch size must be smaller than number of "
                             "datapoints")
        self._batch_size = batch_size
        self._classification = classification
        self._loss = Loss()
        if weighted:
            self._partitioner = WeightedPartitioner(len(data), batch_size,
                                                    data)
        else:
            self._partitioner = Partitioner(len(data), batch_size)
        self._plotter = Plotter()

    def forward_pass_one_batch(self, batch_ids: List[int]) -> float:
        """Performs the forward pass for one batch of the data.

        Parameters
        ----------
        batch_ids : List[int]
            The random list of ids for the current batch

        Returns
        -------
        float
            The total loss of the batch (to keep track)
        """
        total_loss = 0
        for i in batch_ids:
            labelled_point = self._data.loc[i].to_numpy()
            x, y = labelled_point[:-2], int(labelled_point[-2])
            # Do the forward pass and save the predicted value to the df
            y_hat = self._network.forward_pass_one_datapoint(x)
            total_loss += self._loss(y_hat, y)
            self._data.at[i, 'y_hat'] = max(range(len(y_hat)),
                                            key=y_hat.__getitem__)
            self.store_gradients(i)
        # Return the total loss for this batch
        return total_loss

    def run(self):
        """Performs training/validation/testing
        """
        raise NotImplementedError("Cannot call from base class")

    def store_gradients(self, batch_id):
        pass

    def abs_generate_scatter(self, phase: str = 'training', title: str = ''):
        """Creates scatter plot from the data and their predicted values

        Parameters
        ----------
        phase : str
            The phase of learning
        title : str
            An optional title to append to the plot
        """
        self._plotter.plot_predictions(self._data, phase, title)
