import math
from typing import List


class Softmax:
    """Class to represent the softmax function
    """

    def __init__(self):
        """Constructor method
        """
        self._normalisation = 1.0
        self._max_z = 0.0

    def normalise(self, z: List[float]):
        """Calculates the normalisation constant for the softmax function. We
        wish to avoid any overflow errors, so we multiply the normalisation
        constant by :math:`e^{-m}`. We account for this when finding the
        Softmax value later.

        Parameters
        ----------
        z : List[float]
            The vector of values from the output layer of the main network
        """
        self._max_z = max(z)
        self._normalisation = sum([math.exp(z_i - self._max_z) for z_i in z])

    def __call__(self, z_k: float) -> float:
        """The softmax function. Note we multiply top and bottom by _max_z to
        avoid any overflow error.

        Parameters
        ----------
        z_k : float
            The value of an output neuron

        Returns
        -------
        float
            The softmax output
        """
        return math.exp(z_k - self._max_z) / self._normalisation
