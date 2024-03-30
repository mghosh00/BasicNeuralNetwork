import math
import pandas as pd

from neural_network.components import Network

from .abstract_learner import AbstractLearner


class Validator(AbstractLearner):
    """Class to validate a neural network
    """

    def __init__(self, network: Network, data: pd.DataFrame, batch_size: int,
                 weighted: bool = False):
        """Constructor method

        Parameters
        ----------
        network : Network
            The neural network
        data : pd.DataFrame
            All the validation data for the `Network`
        batch_size : int
            The number of datapoints used in each epoch
        weighted : bool
            If `True` then we use the WeightedPartitioner, otherwise we use
            the standard Partitioner
        """
        super().__init__(network, data, batch_size, weighted)
        self._epoch = 0

    def validate(self, factor: int):
        """Performs validation of the network.

        Parameters
        ----------
        factor : int
            The epochs on which we need to print out the validation
        """
        self._epoch += 1
        total_loss = 0
        batch_partition = self._partitioner()
        for iteration in range(math.ceil(len(self._data)
                                         / self._batch_size)):
            batch_ids = batch_partition[iteration]
            total_loss += self.forward_pass_one_batch(batch_ids)
        loss = round(total_loss / len(self._data), 8)
        if self._epoch % factor == 0:
            print(f"Validation loss: {loss}")

        if not self._regression:
            self._update_categorical_dataframe()

        return loss

    def generate_scatter(self, title: str = ''):
        """Creates scatter plot from the data and their predicted values

        Parameters
        ----------
        title : str
            An optional title to append to the plot
        """
        super().abs_generate_scatter(phase='validation', title=title)

    def comparison_scatter(self, title: str = ''):
        """Creates scatter plot comparing predicted to actual values (for
        regressional problems only).

        Parameters
        ----------
        title : str
            An optional title to append to the plot
        """
        super().abs_comparison_scatter(phase='validation', title=title)
