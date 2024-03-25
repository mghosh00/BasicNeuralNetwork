import math
from typing import List


class CrossEntropyLoss:
    """Class to represent the cross entropy loss function for classification
    networks.
    """

    def __init__(self):
        """Constructor method
        """
        pass

    def __call__(self, y_hat: List[float], y: int) -> float:
        """The loss function.

        Parameters
        ----------
        y_hat : List[float]
            Output vector from softmax layer
        y : int
            Target class (in {0, 1, ...})

        Returns
        -------
        float
            Cross entropy loss value
        """
        softmax_value = y_hat[y]

        # This should be a probability
        if 0 <= softmax_value <= 1:
            return - math.log(y_hat[y])

        else:
            raise ValueError(f"Softmax value should be between 0 and 1"
                             f" (y_hat[{y}] = {softmax_value})")
