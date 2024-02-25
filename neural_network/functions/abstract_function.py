import typing

class AbstractFunction:
    """Class to represent an abstract function
    """

    def __init__(self):
        """Constructor method
        """
        pass

    def __call__(self, x: float | typing.List, w: typing.List = None) -> float:
        """Calling of the function

        Parameters
        ----------
        x : float | typing.List
            The input value
        w : typing.List [Optional, Default = None]
            The weights
        Returns
        -------
        float
            The output value
        """
        raise NotImplementedError("Cannot call from base class")

    def gradient(self, x: float | typing.List, w: typing.List = None)\
            -> float | typing.List:
        """The gradient of the function

        Parameters
        ----------
        x : float | typing.List
            The input value
        w : typing.List
            The weights

        Returns
        -------
        float | typing.List
            The gradient of the function
        """
        raise NotImplementedError("Cannot call from base class")
