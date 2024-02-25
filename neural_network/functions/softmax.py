import math
from typing import List


class Softmax:
    """Class to represent the softmax function
    """

    def __init__(self):
        """Constructor method
        """
        self._normalisation = 1

    def normalisation(self, z: List[float]):
        """The normalisation constant for the softmax function

        Parameters
        ----------
        z : List[float]
            The vector of values from the output layer of the main network

        Returns
        -------
        float
            The normalisation constant
        """
        self._normalisation =  sum([math.exp(z_i) for z_i in z])

    def __call__(self, z_k: float) -> float:
        """The loss function

        Parameters
        ----------
        z_k : float
            The value of an output node

        Returns
        -------
        float
            The softmax output
        """
        return math.exp(z_k) / self._normalisation
