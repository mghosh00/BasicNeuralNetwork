import math
import typing


class Loss:
    """Class to represent the cross entropy loss function
    """

    def __init__(self):
        """Constructor method
        """
        pass

    def __call__(self, y_hat: typing.List, y: int) -> float:
        """The loss function

        Parameters
        ----------
        y_hat : typing.List
            Output vector from softmax layer
        y : int
            Target class (in {0, 1, ...})

        Returns
        -------
        float
            Loss value
        """
        return - math.log(y_hat[y])
