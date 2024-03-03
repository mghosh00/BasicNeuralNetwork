from typing import Callable, Any
from inspect import signature

import numpy as np
import pandas as pd


class AbstractDataGenerator:
    """Class to randomly generate datapoints and categorise them according to
    a given rule.
    """

    def __init__(self, classifier: Callable[[float, ...], Any],
                 num_datapoints: int):
        """Constructor method

        Parameters
        ----------
        classifier : Callable[[float, ...], Any]
            A rule which takes a certain number of coordinates and returns a
            value representing the class of the datapoint
        num_datapoints : int
            The number of datapoints to be generated
        """
        self._classifier = classifier
        dimensions = len(signature(classifier).parameters)
        if dimensions < 1:
            raise ValueError(f"classifier must have at least one coordinate "
                             f"(num_coordinates = {dimensions})")
        if num_datapoints < 1:
            raise ValueError(f"Must have at least one datapoint "
                             f"(num_datapoints = {num_datapoints})")
        self._dimensions = dimensions
        self._num_datapoints = num_datapoints
        self._df = pd.DataFrame(columns=[f"x_{i + 1}"
                                         for i in range(dimensions)] + ['y'])
        self._x = []

    def _generate_data(self):
        """To generate the data - cannot be called from AbstractDataGenerator
        """
        raise NotImplementedError("Cannot call generate_data or __call__ "
                                  "from base class")

    def __call__(self) -> pd.DataFrame:
        """Writes to self._df with the generated data and classes.

        Returns
        -------
        pd.DataFrame
            self._df with the newly generated data
        """
        # Generates data (using a subclass)
        self._generate_data()

        # The below is a dictionary containing categories as keys and lists of
        # datapoint indices as values
        categories = {}
        for j in range(self._num_datapoints):
            # The below will evaluate the classifier for one datapoint x_j
            # using all its coordinates as inputs
            category = self._classifier(*[self._x[i][j]
                                          for i in range(self._dimensions)])

            # If this is the first occurrence of the category, we create a new
            # list of indices. Else, we append this index to the current list.
            if category in categories.keys():
                categories[category].append(j)
            else:
                categories[category] = [j]

        # Now we use labels 0 to (num_classes - 1) to standardise
        y = [0] * self._num_datapoints
        for k, category in enumerate(categories.keys()):
            for j in categories[category]:
                y[j] = k

        # Finally, update the df
        for i in range(self._dimensions):
            self._df[f'x_{i}'] = self._x[i]
        self._df['y'] = np.array(y)
        return self._df

    def write_to_csv(self, title: str, directory: str = ''):
        """Writes the generated data to a .csv file.

        Parameters
        ----------
        title : str
            The title for the .csv file
        directory : str (Optional, Default = '')
            The directory for the file
        """
        path = f"{directory}/{title}.csv" if directory else f"{title}.csv"
        self._df.to_csv(path)
