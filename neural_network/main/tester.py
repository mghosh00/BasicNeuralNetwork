import math
import statistics

import pandas as pd

from neural_network.components import Network

from .abstract_simulator import AbstractSimulator


class Tester(AbstractSimulator):
    """Class to test a neural network.
    """

    def __init__(self, network: Network, data: pd.DataFrame, batch_size: int,
                 weighted: bool = False, classification: bool = True):
        """Constructor method

        Parameters
        ----------
        network : Network
            The neural network
        data : pd.DataFrame
            All the testing data for the `Network`
        batch_size : int
            The number of datapoints used in each epoch
        weighted : bool
            If `True` then we use the WeightedPartitioner, otherwise we use
            the standard Partitioner
        classification : bool
            If `True` then we are classifying, otherwise it will be regression
        """
        super().__init__(network, data, batch_size, weighted, classification)

    def run(self):
        """Performs testing of the network.
        """
        total_loss = 0
        batch_partition = self._partitioner()
        for iteration in range(math.ceil(len(self._data)
                                         / self._batch_size)):
            batch_ids = batch_partition[iteration]
            total_loss += self.forward_pass_one_batch(batch_ids)
        loss = round(total_loss / len(self._data), 8)
        print(f"Testing loss: {loss}")

    def generate_scatter(self, title: str = ''):
        """Creates scatter plot from the data and their predicted values.

        Parameters
        ----------
        title : str
            An optional title to append to the plot
        """
        super().abs_generate_scatter(phase='testing', title=title)

    def generate_confusion(self):
        """Creates a confusion matrix from the results.
        """
        # num_classes = len(set(self._data['y'].to_numpy()))
        # confusion_df = pd.DataFrame(index=range(num_classes),
        #                             columns=range(num_classes))
        # for i in range(len(self._data)):
        #     actual = int(self._data.at[i, 'y'])
        #     predicted = int(self._data.at[i, 'y_hat'])
        #     confusion_df.at[actual, predicted] += 1
        confusion_df = pd.crosstab(self._data.y, self._data.y_hat)
        print("Confusion matrix for testing data:")
        print(confusion_df)
        num_classes = len(confusion_df)
        dice_scores = {}
        for i in range(num_classes):
            true_positive = confusion_df.at[i, i]
            sum_row = sum(confusion_df.iloc[i, :])
            sum_column = sum(confusion_df.iloc[:, i])
            dice = 2 * true_positive / (sum_row + sum_column)
            dice_scores[i] = dice
        print(f"Dice scores: {dice_scores}")
        print(f"Mean dice score: {statistics.mean(dice_scores.values())}")
