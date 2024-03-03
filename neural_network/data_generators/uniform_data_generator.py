from typing import List, Callable, Any

import numpy as np

from .abstract_data_generator import AbstractDataGenerator


class UniformDataGenerator(AbstractDataGenerator):
    """Class to randomly generate datapoints and categorise them according to
    a given rule, with data being generated via a uniform distribution.
    """

    def __init__(self, classifier: Callable[[float, ...], Any],
                 num_datapoints: int, lower_bounds: List[float],
                 upper_bounds: List[float]):
        """Constructor method

        Parameters
        ----------
        classifier : Callable[[float, ...], Any]
            A rule which takes a certain number of coordinates and returns a
            value representing the class of the datapoint
        num_datapoints : int
            The number of datapoints to be generated
        lower_bounds : List[float]
            A list of lower bounds for each coordinate
        upper_bounds : List[float]
            A list of upper bounds for each coordinate
        """
        super().__init__(classifier, num_datapoints)
        if len(lower_bounds) != self._dimensions:
            raise ValueError(f"The classifier method accepts "
                             f"{self._dimensions} parameters but we have "
                             f"{len(lower_bounds)} lower bounds.")
        if len(upper_bounds) != self._dimensions:
            raise ValueError(f"The classifier method accepts "
                             f"{self._dimensions} parameters but we have "
                             f"{len(upper_bounds)} upper bounds.")
        for i in range(len(lower_bounds)):
            if lower_bounds[i] > upper_bounds[i]:
                raise ValueError(f"All lower bounds must be lower than their "
                                 f"related upper bounds ({lower_bounds[i]}"
                                 f" > {upper_bounds[i]})")
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def _generate_data(self):
        """To generate uniformly distributed data
        """
        # A list of numpy arrays, each representing one observed coordinate for
        # all datapoints
        x = [np.random.uniform(low=self._lower_bounds[i],
                               high=self._upper_bounds[i],
                               size=self._num_datapoints)
             for i in range(self._dimensions)]
        self._x = x
