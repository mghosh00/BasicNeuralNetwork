from typing import Union, List, Callable, Any

import numpy as np

from .abstract_data_generator import AbstractDataGenerator


class NormalDataGenerator(AbstractDataGenerator):
    """Class to randomly generate datapoints and categorise them according to
    a given rule, with data being generated via a normal distribution.
    """

    custom_type = Union[
        Callable[[float], Any],
        Callable[[float, float], Any],
        Callable[[float, float, float], Any],
        Callable[[float, float, float, float], Any]
    ]

    def __init__(self, classifier: custom_type,
                 num_datapoints: int, means: List[float],
                 std_devs: List[float]):
        """Constructor method

        Parameters
        ----------
        classifier : custom_type
            A rule which takes a certain number of coordinates and returns a
            value representing the class of the datapoint
        num_datapoints : int
            The number of datapoints to be generated
        means : List[float]
            A list of means for each coordinate
        std_devs : List[float]
            A list of standard deviations for each coordinate
        """
        super().__init__(classifier, num_datapoints)
        if len(means) != self._dimensions:
            raise ValueError(f"The classifier method accepts "
                             f"{self._dimensions} parameters but we have "
                             f"{len(means)} means.")
        if len(std_devs) != self._dimensions:
            raise ValueError(f"The classifier method accepts "
                             f"{self._dimensions} parameters but we have "
                             f"{len(std_devs)} standard deviations.")
        for i in range(len(std_devs)):
            if std_devs[i] <= 0:
                raise ValueError(f"All standard deviations must be positive "
                                 f"({std_devs[i]} <= 0)")
        self._means = means
        self._std_devs = std_devs

    def _generate_data(self):
        """To generate normally distributed data
        """
        # A list of numpy arrays, each representing one observed coordinate for
        # all datapoints
        x = [np.random.normal(loc=self._means[i], scale=self._std_devs[i],
                              size=self._num_datapoints)
             for i in range(self._dimensions)]
        self._x = x
