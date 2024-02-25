import typing

from .abstract_function import AbstractFunction


class TransferFunction(AbstractFunction):
    """Class to represent the transfer function
    """

    def __init__(self):
        """Constructor method
        """
        super().__init__()

    def __call__(self, o: typing.List, w: typing.List = None) -> float:
        """Implementation of the transfer function

        Parameters
        ----------
        o : typing.List
            Output of previous layer
        w : typing.List
            Weights and bias

        Returns
        -------
        float
            Output to function
        """
        o_size = len(o)
        if len(w) != o_size + 1:
            raise ValueError(f"w and o do not match in size ({len(w)} != "
                             f"{o_size + 1})")
        return sum([o[i] * w[i] for i in range(o_size)]) + w[o_size]

    def gradient(self, o: typing.List, w: typing.List = None) -> typing.List:
        """Gradient of transfer

        Parameters
        ----------
        o : typing.List
            Output of previous layer
        w : typing.List
            Weights

        Returns
        -------
        typing.List
            Gradient of transfer function w.r.t w (including the bias)
        """
        return o + [0]
