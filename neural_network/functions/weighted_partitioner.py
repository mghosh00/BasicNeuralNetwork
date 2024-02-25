import math
import random
import typing
import pandas as pd

from .partitioner import Partitioner


class WeightedPartitioner(Partitioner):
    """Class to create `m` sets from a list of `n` integers weighted by which
    ground truth class each integer lies in
    """

    def __init__(self, n: int, m: int, df: pd.DataFrame):
        """Constructor method

        Parameters
        ----------
        n : int
            Number of integers
        m : int
            Number of sets for the partition
        df : pd.DataFrame
            The classes for the integers
        """
        super().__init__(n, m)
        if not len(df) == n:
            raise ValueError(f"n must equal the length of the dataframe "
                             f"(n = {n}, len(df) = {len(df)})")
        self._num_classes = len(set(df['y'].to_numpy()))
        self._class_dict = {j: [] for j in range(self._num_classes)}
        for i in range(len(df)):
            self._class_dict[int(df.loc[i, 'y'])].append(i)

    def __call__(self) -> typing.List:
        """Uses weights for each class to create sets of size `m` containing
        integers (sampled with replacement).

        Returns
        -------
        typing.List
            The list of sets
        """
        # Produces a list of `n` class indices
        chosen_classes = random.choices(population=range(self._num_classes),
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
