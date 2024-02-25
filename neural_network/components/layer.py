from typing import List

from .neuron import Neuron


class Layer:
    """Class to represent one `Layer` of a network
    """

    def __init__(self, _id: int, num_neurons: int):
        """Constructor method

        Parameters
        ----------
        _id : int
            The id of the `Layer`
        num_neurons : int
            The number of `Neurons` in the `Layer`
        """
        self._id = _id
        self._num_neurons = num_neurons
        self._neurons = [Neuron(_id, j) for j in range(num_neurons)]

    def get_id(self) -> int:
        """Getter method

        Returns
        -------
        int
            The id
        """
        return self._id

    def get_neurons(self) -> List[Neuron]:
        """Getter method

        Returns
        -------
        List[Neuron]
            The `Neurons` in the `Layer`
        """
        return self._neurons

    def __str__(self):
        return f"Layer {self._id}"
