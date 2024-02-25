import math
import typing

from .abstract_function import AbstractFunction


class Sigmoid(AbstractFunction):
    """Class to represent the Sigmoid function
    """

    def __init__(self):
        """Constructor method
        """
        super().__init__()

    def __call__(self, x: float, w: typing.List = None) -> float:
        """Implementation of Sigmoid

        Parameters
        ----------
        x : float
            Input to function
        w : typing.List
            Weights (not used here)

        Returns
        -------
        float
            Output to function
        """
        return 1 / (1 + math.exp(x))

    def gradient(self, x: float, w: typing.List = None) -> float:
        """Gradient of Sigmoid

        Parameters
        ----------
        x : float
            Input to function
        w : typing.List
            Not used for this class

        Returns
        -------
        float
            Gradient of Sigmoid
        """
        return self(x) * (1 - self(x))
