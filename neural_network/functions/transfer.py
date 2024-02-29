from typing import List

from .abstract_function import AbstractFunction


class TransferFunction(AbstractFunction):
    """Class to represent the transfer function
    """

    def __init__(self):
        """Constructor method
        """
        super().__init__()

    def __call__(self, o: List[float], w: List[float] = None) -> float:
        """Implementation of the transfer function

        Parameters
        ----------
        o : List[float]
            Output of previous layer
        w : List[float]
            Weights and bias

        Returns
        -------
        float
            Output to function
        """
        if w is None:
            raise ValueError("w cannot be None")
        o_size = len(o)
        if len(w) != o_size + 1:
            raise ValueError(f"w is not one element longer than o "
                             f"({len(w)} != {o_size} + 1)")
        return sum([o[i] * w[i] for i in range(o_size)]) + w[o_size]

    def gradient(self, o: List[float], w: List[float] = None) -> List[float]:
        """Gradient of transfer

        Parameters
        ----------
        o : List[float]
            Output of previous layer
        w : List[float]
            Weights

        Returns
        -------
        List[float]
            Gradient of transfer function w.r.t w (including the bias)
        """
        return o + [0]
