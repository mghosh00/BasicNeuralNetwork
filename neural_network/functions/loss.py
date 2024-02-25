import math
from typing import List


class Loss:
    """Class to represent the cross entropy loss function
    """

    def __init__(self):
        """Constructor method
        """
        pass

    def __call__(self, y_hat: List[float], y: int) -> float:
        """The loss function

        Parameters
        ----------
        y_hat : List[float]
            Output vector from softmax layer
        y : int
            Target class (in {0, 1, ...})

        Returns
        -------
        float
            Loss value
        """
        return - math.log(y_hat[y])
