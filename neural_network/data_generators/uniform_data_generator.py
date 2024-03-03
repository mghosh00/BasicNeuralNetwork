from typing import List, Tuple, Callable, Any

import numpy as np

from .abstract_data_generator import AbstractDataGenerator


class UniformDataGenerator(AbstractDataGenerator):
    """Class to randomly generate datapoints and categorise them according to
    a given rule, with data being generated via a uniform distribution.
    """

    def __init__(self, classifier: Callable[[float, ...], Any],
                 num_datapoints: int, bounds: List[Tuple[float, float]]):
        """Constructor method

        Parameters
        ----------
        classifier : Callable[[float, ...], Any]
            A rule which takes a certain number of coordinates and returns a
            value representing the class of the datapoint
        num_datapoints : int
            The number of datapoints to be generated
        bounds : List[Tuple[float, float]]
            A list of lower, upper bound pairs for each coordinate
        """
        super().__init__(classifier, num_datapoints)
        for bound_pair in bounds:
            if len(bound_pair) != 2:
                raise ValueError(f"Each bound pair in bounds must have a "
                                 f"length of 2 (error: {bound_pair}).")
            if bound_pair[0] >= bound_pair[1]:
                raise ValueError(f"In each bound pair in bounds, the first "
                                 f"element is the lower bound and the second "
                                 f"is the upper bound (error: {bound_pair}).")
        if len(bounds) != self._dimensions:
            raise ValueError(f"The classifier method accepts "
                             f"{self._dimensions} parameters but we have "
                             f"{len(bounds)} bound pairs in bounds list.")
        self._bounds = bounds

    def _generate_data(self):
        """To generate uniformly distributed data
        """
        # A list of numpy arrays, each representing one observed coordinate for
        # all datapoints
        x = [np.random.uniform(low=self._bounds[i][0], high=self._bounds[i][1],
                               size=self._num_datapoints)
             for i in range(self._dimensions)]
        self._x = x
