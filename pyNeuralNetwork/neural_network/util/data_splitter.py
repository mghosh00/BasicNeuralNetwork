import math
from typing import List, Tuple
import pandas as pd


class DataSplitter:
    """Class to split a dataset into training, validation and testing, given a
    split ratio.
    """

    def __init__(self, path: str, proportions: List[int]):
        """Constructor method

        Parameters
        ----------
        path : str
            Path to the .csv file containing the data
        proportions : typing.List
            The proportions in the sequence training:validation:testing
        """
        self._df = pd.read_csv(path, index_col=0)
        if not 1 <= len(proportions) <= 3:
            raise ValueError("proportions must have 1-3 elements denoting the"
                             "train:validation:test ratio")
        self._proportions = proportions

    def split(self) -> Tuple[pd.DataFrame, ...]:
        """Main method for the class - splits the data into train:valid:test

        Returns
        -------
        Tuple[pd.DataFrame, ...]
            A tuple containing the training, validation and testing dataframes
            or fewer, if fewer proportions have been passed
        """
        n = len(self._df)
        prop_total = sum(self._proportions)
        splits = [0]
        dfs = []
        for i in range(len(self._proportions) - 1):
            len_new_df = math.floor(n * (self._proportions[i] / prop_total))
            if len_new_df == 0:
                len_new_df = 1
            splits.append(sum(splits) + len_new_df)
            new_df = self._df.iloc[splits[i]:splits[i + 1]]
            new_df.index = range(len_new_df)
            dfs.append(new_df)

        len_final_df = n - splits[-1]
        final_df = self._df.iloc[splits[-1]:]
        final_df.index = range(len_final_df)
        dfs.append(final_df)

        return tuple(dfs)
