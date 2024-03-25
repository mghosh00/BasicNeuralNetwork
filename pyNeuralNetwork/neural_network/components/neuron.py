from typing import Tuple


class Neuron:
    """Class to represent a single `Neuron` in a neural network
    """

    def __init__(self, layer_id: int, row_id: int):
        """Constructor method

        Parameters
        ----------
        layer_id : int
            The layer of the network
        row_id : int
            The row in the layer
        """
        self._id = layer_id, row_id
        self._bias = 0.0
        self._value = None
        self.bias_gradients = []

    def set_bias(self, bias: float):
        """Setter method

        Parameters
        ----------
        bias : float
            The new bias
        """
        self._bias = bias

    def get_bias(self) -> float:
        """Getter method

        Returns
        -------
        float
            The bias
        """
        return self._bias

    def set_value(self, value: float):
        """Setter method

        Parameters
        ----------
        value : float
            The new value
        """
        self._value = value

    def get_value(self) -> float:
        """Getter method

        Returns
        -------
        float
            The value
        """
        return self._value

    def get_id(self) -> Tuple[int, ...]:
        """Getter method

        Returns
        -------
        Tuple[int, ...]
            The id
        """
        return self._id

    def __str__(self):
        return f"Neuron {self._id}"
