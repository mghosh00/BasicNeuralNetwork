from typing import List


class AbstractFunction:
    """Class to represent an abstract function
    """

    def __init__(self):
        """Constructor method
        """
        pass

    def __call__(self, x: float | List[float], w: List[float] = None) -> float:
        """Calling of the function

        Parameters
        ----------
        x : float | List[float]
            The input value
        w : List[float] [Optional, Default = None]
            The weights
        Returns
        -------
        float
            The output value
        """
        raise NotImplementedError("Cannot call from base class")

    def gradient(self, x: float | List[float], w: List[float] = None)\
            -> float | List[float]:
        """The gradient of the function

        Parameters
        ----------
        x : float | List[float]
            The input value
        w : List[float]
            The weights

        Returns
        -------
        float | List[float]
            The gradient of the function
        """
        raise NotImplementedError("Cannot call from base class")
