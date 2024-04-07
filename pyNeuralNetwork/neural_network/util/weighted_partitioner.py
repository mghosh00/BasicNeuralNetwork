import math
import random
from typing import List
import pandas as pd

from .partitioner import Partitioner


class WeightedPartitioner(Partitioner):
    """Class to create `m` sets from a list of `n` integers weighted by which
    ground truth class each integer lies in.
    """

    def __init__(self, n: int, m: int, df: pd.DataFrame,
                 do_regression: bool = False, num_bins: int = 10):
        """Constructor method

        Parameters
        ----------
        n : int
            Number of integers
        m : int
            Number of sets for the partition
        df : pd.DataFrame
            The classes for the integers
        do_regression : bool
            Whether we are partitioning regressional or classificational data
        num_bins : int
            If regression is True, this represents the number of bins to
            split the data in to. Otherwise, this parameter is ignored
        """
        super().__init__(n, m)
        if not len(df) == n:
            raise ValueError(f"n must equal the length of the dataframe "
                             f"(n = {n}, len(df) = {len(df)})")
        if do_regression:
            self._num_bins = num_bins
            class_dict = {j: [] for j in range(self._num_bins)}
            min_y, max_y = min(df['y']), max(df['y'])
            class_width = (max_y - min_y) / num_bins
            for i in range(len(df)):
                y = df.loc[i, 'y']
                chosen_bin = int((y - min_y) / class_width)
                if chosen_bin == self._num_bins:
                    chosen_bin = self._num_bins - 1
                class_dict[chosen_bin].append(i)

            # Need to account for potentially empty class lists
            for j in range(self._num_bins):
                if not class_dict[j]:
                    class_dict.pop(j)
                    self._num_bins -= 1

            self._class_dict = {}
            for i, j in enumerate(class_dict.keys()):
                # Relabelling the classes
                self._class_dict[i] = class_dict[j]
        else:
            self._num_bins = len(set(df['y'].to_numpy()))
            self._class_dict = {j: [] for j in range(self._num_bins)}
            for i in range(len(df)):
                self._class_dict[int(df.loc[i, 'y'])].append(i)

    def __call__(self) -> List[List[int]]:
        """Uses weights for each class to create sets of size `m` containing
        integers (sampled with replacement).

        Returns
        -------
        List[List[int]]
            The list of sets
        """
        # Produces a list of `n` class indices
        chosen_classes = random.choices(population=range(self._num_bins),
                                        k=self._n)
        output_list = []
        num_sets = math.ceil(self._n / self._m)
        for i in range(num_sets - 1):
            inner_list = []
            for j in range(self._m):
                chosen_class = chosen_classes[i * self._m + j]
                inner_list.append(random.choice(
                    self._class_dict[chosen_class]))
            output_list.append(inner_list)
        inner_list = []
        for k in range((num_sets - 1) * self._m, self._n):
            chosen_class = chosen_classes[k]
            inner_list.append(random.choice(self._class_dict[chosen_class]))
        output_list.append(inner_list)
        return output_list
