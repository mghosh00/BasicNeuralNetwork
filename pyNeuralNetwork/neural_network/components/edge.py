import random
from typing import Tuple

from .neuron import Neuron


class Edge:
    """Class to represent an `Edge` joining two `Nodes` of a `Network`
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
        left_id = left_neuron.get_id()[0]
        right_id = right_neuron.get_id()[0]
        if left_id + 1 != right_id:
            raise ValueError(f"Edge must connect adjacent layers "
                             f"(left: {left_id}, right: {right_id})")
        self._id = (left_neuron.get_id()[0], left_neuron.get_id()[1],
                    right_neuron.get_id()[1])
        self._weight = random.uniform(-1, 1)
        self.loss_gradients = []
        self._delta = 0.0
        self._velocity = 0.0

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

    def get_id(self) -> Tuple[int, ...]:
        """Getter method

        Returns
        -------
        Tuple[int, ...]
            The id
        """
        return self._id

    def set_delta(self, delta: float):
        """Setter method

        Parameters
        ----------
        delta : float
            The new delta
        """
        self._delta = delta

    def get_delta(self) -> float:
        """Getter method

        Returns
        -------
        float
            The delta
        """
        return self._delta

    def set_velocity(self, velocity: float):
        """Setter method

        Parameters
        ----------
        velocity : float
            The new velocity
        """
        self._velocity = velocity

    def get_velocity(self) -> float:
        """Getter method

        Returns
        -------
        float
            The velocity
        """
        return self._velocity

    def __str__(self):
        return f"Edge between {self._left_neuron} and {self._right_neuron}"
