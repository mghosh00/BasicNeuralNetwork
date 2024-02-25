import typing

from .abstract_function import AbstractFunction


class ReLU(AbstractFunction):
    """Class to represent the ReLU function
    """

    def __init__(self, leak: float = 0.0):
        """Constructor method

        Parameters
        ----------
        leak : float
            The parameter to be used if this is a LeakyReLU
        """
        super().__init__()
        self._leak = leak

    def __call__(self, x: float, w: typing.List = None) -> float:
        """Implementation of ReLU

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
        return x if x >= 0 else x * self._leak

    def gradient(self, x: float, w: typing.List = None) -> float:
        """Gradient of ReLU

        Parameters
        ----------
        x : float
            Input to function
        w : typing.List
            Not used for this class

        Returns
        -------
        float
            Gradient of ReLU
        """
        return 1 if x >= 0 else self._leak
