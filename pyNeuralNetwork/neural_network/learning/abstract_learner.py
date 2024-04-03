from typing import List
import pandas as pd

from neural_network.util import Partitioner
from neural_network.util import WeightedPartitioner
from neural_network.functions import CrossEntropyLoss
from neural_network.functions import MSELoss
from neural_network.components import Network

from .plotter import Plotter


class AbstractLearner:
    """Base class for trainer, tester and validator
    """

    def __init__(self, network: Network, data: pd.DataFrame, batch_size: int,
                 weighted: bool = False, bins: int = 10):
        """Constructor method

        Parameters
        ----------
        network : Network
            The neural network to train
        data : pd.DataFrame
            All the data for the `Network`
        batch_size : int
            The number of datapoints per batch for an epoch
        weighted : bool
            If `True` then we use the `WeightedPartitioner`, otherwise we use
            the standard `Partitioner`
        bins : int
            If `weighted` is `True` and `regression` is `True`, then we need
            to specify the number of bins for the weighted partitioner
        """
        self._network = network
        self._regression = network.is_regressor()
        data = data.copy()
        # Ensure that number of input neurons equals number of features
        n = len(data.columns) - 1
        m = network.get_neuron_counts()[0]
        if m != n:
            raise ValueError(f"Number of features must match number of "
                             f"initial neurons (features = {n}, initial "
                             f"neurons = {m})")

        # Renaming of columns
        data.columns = [f'x_{i + 1}' for i in range(n)] + ['y']

        # Ensure that batch_size is not too big
        if batch_size > len(data):
            raise ValueError("Batch size must be smaller than number of "
                             "datapoints")
        self._batch_size = batch_size

        if not self._regression:
            # Save the category names to be used for plots and output data
            self._category_names = sorted(list(set(data['y'])))

            # Ensure that number of network output neurons equals the number of
            # classes in the dataframe
            num_classes = len(self._category_names)
            num_outputs = network.get_neuron_counts()[-1]
            if num_outputs < num_classes:
                raise ValueError(f"The number of output neurons in the "
                                 f"network ({num_outputs}) is less than the "
                                 f"number of classes in the dataframe "
                                 f"({num_classes})")

            # Change the category names to integers from 0 to
            # num_classes - 1 for the numerical calculations, but save the
            # category names for reference in plots.
            data['y_hat'] = [0] * len(data)
            self._categorical_data = data
            numerical_data = data.replace({'y': {self._category_names[i]: i
                                                 for i in range(num_classes)}})
            self._data = numerical_data
            self._cross_entropy_loss = CrossEntropyLoss()
        else:
            # If we are doing regression, we have no categories, and we will
            # use a mean squared error loss
            data['y_hat'] = [0.0] * len(data)
            self._data = data
            self._mse_loss = MSELoss()

        if weighted:
            self._partitioner = WeightedPartitioner(len(self._data),
                                                    batch_size, self._data,
                                                    self._regression,
                                                    num_bins=bins)
        else:
            self._partitioner = Partitioner(len(self._data), batch_size)

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
            x, y = labelled_point[:-2], labelled_point[-2]

            # Do the forward pass and save the predicted value to the df
            if self._regression:
                pred_value = self._network.forward_pass_one_datapoint(x)[0]
                total_loss += self._mse_loss(pred_value, y)
                self._data.at[i, 'y_hat'] = pred_value
            else:
                # We choose the class with maximal softmax probability as our
                # y_hat for output
                y = int(y)
                softmax_vector = self._network.forward_pass_one_datapoint(x)
                total_loss += self._cross_entropy_loss(softmax_vector, y)
                self._data.at[i, 'y_hat'] = max(range(len(softmax_vector)),
                                                key=softmax_vector.__getitem__)
            self.store_gradients(i)
        # Return the total loss for this batch
        return total_loss

    def run(self):
        """Performs training/validation/testing
        """
        raise NotImplementedError("Cannot call from base class")

    def store_gradients(self, batch_id: int):
        """To be overridden by subclasses.

        Parameters
        ----------
        batch_id : id of the current batch
        """
        return

    def _update_categorical_dataframe(self):
        """Update the categorical dataframe with y_hat data but using the
        original categories from the data - to be used for plotting and
        outputs to the user. Note that this method will be called after
        training/testing/validation is complete so that the y_hat values are
        fully updated.
        """
        if self._regression:
            raise RuntimeError("Cannot call _update_categorical_dataframe"
                               "with a regressional network")
        names = self._category_names
        self._categorical_data['y_hat'] = \
            list(self._data.replace({'y_hat':
                                    {i: names[i] for i in range(len(names))}})
                 ['y_hat'])

    def abs_generate_scatter(self, phase: str = 'training', title: str = ''):
        """Creates scatter plot from the data and their predicted values. For
        classification, we use the categories the user provided with the data
        instead of arbitrary integer classes.

        Parameters
        ----------
        phase : str
            The phase of learning
        title : str
            An optional title to append to the plot
        """
        if self._regression:
            Plotter.datapoint_scatter(self._data, phase, title,
                                      regression=True)
        else:
            Plotter.datapoint_scatter(self._categorical_data, phase, title)

    def abs_comparison_scatter(self, phase: str = 'training', title: str = ''):
        """Creates scatter plot comparing the predicted and actual values in
        a regressional problem. Cannot be called with classification problems

        Parameters
        ----------
        phase : str
            The phase of learning
        title : str
            An optional title to append to the plot
        """
        if self._regression:
            Plotter.comparison_scatter(self._data, phase, title)
        else:
            raise RuntimeError("Cannot call this method with categorical data")
