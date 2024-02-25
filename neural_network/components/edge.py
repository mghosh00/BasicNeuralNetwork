import random
import typing

from .neuron import Neuron


class Edge:
    """Class to represent an `Edge` joining two `Nodes` of a network
    """

    def __init__(self, left_neuron: Neuron, right_neuron: Neuron):
        """Constructor method

        Parameters
        ----------
        left_neuron : Neuron
            The left `Neuron` of the edge
        right_neuron : Neuron
            The right `Neuron` of the edge
        """
        self._left_neuron = left_neuron
        self._right_neuron = right_neuron
        self._id = (left_neuron.get_id()[0], left_neuron.get_id()[1],
                    right_neuron.get_id()[1])
        self._weight = random.random()
        self.loss_gradients = []

    def set_weight(self, weight: float):
        """Setter method

        Parameters
        ----------
        weight : float
            The new weight
        """
        self._weight = weight

    def get_weight(self) -> float:
        """Getter method

        Returns
        -------
        float
            The weight
        """
        return self._weight

    def get_left_neuron(self) -> Neuron:
        """Getter method

        Returns
        -------
        Neuron
            The left Neuron
        """
        return self._left_neuron

    def get_right_neuron(self) -> Neuron:
        """Getter method

        Returns
        -------
        float
            The right Neuron
        """
        return self._right_neuron

    def get_id(self) -> typing.Tuple:
        """Getter method

        Returns
        -------
        typing.Tuple
            The id
        """
        return self._id

    def __str__(self):
        return f"Edge between {self._left_neuron} and {self._right_neuron}"
