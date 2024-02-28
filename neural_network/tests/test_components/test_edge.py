import unittest
from unittest import TestCase
from unittest import mock

from neural_network import Neuron
from neural_network import Edge


class TestEdge(TestCase):
    """Tests the `Edge` class.
    """

    @mock.patch('random.uniform')
    def setUp(self, mock_random):
        mock_random.return_value = 0.1
        self.left_neuron = Neuron(4, 3)
        self.right_neuron = Neuron(5, 2)
        self.edge = Edge(self.left_neuron, self.right_neuron)

    def test_construct_erroneous(self):
        with self.assertRaises(ValueError) as ve:
            Edge(self.right_neuron, self.left_neuron)
        self.assertEqual(str(ve.exception), "Edge must connect adjacent "
                                            "layers (left: 5, right: 4)")

    def test_construct(self):
        self.assertEqual(self.left_neuron, self.edge._left_neuron)
        self.assertEqual(self.right_neuron, self.edge._right_neuron)
        self.assertEqual((4, 3, 2), self.edge._id)
        self.assertEqual(0.1, self.edge._weight)
        self.assertEqual([], self.edge.loss_gradients)
        self.assertEqual(0.0, self.edge._delta)
        self.assertEqual(0.0, self.edge._velocity)

    def test_weight_getter_and_setter(self):
        self.assertEqual(0.1, self.edge.get_weight())
        self.edge.set_weight(0.4)
        self.assertEqual(0.4, self.edge._weight)

    def test_velocity_getter_and_setter(self):
        self.assertEqual(0.0, self.edge.get_velocity())
        self.edge.set_velocity(0.5)
        self.assertEqual(0.5, self.edge._velocity)

    def test_delta_getter_and_setter(self):
        self.assertEqual(0.0, self.edge.get_delta())
        self.edge.set_delta(0.6)
        self.assertEqual(0.6, self.edge._delta)

    def test_get_id(self):
        self.assertEqual((4, 3, 2), self.edge.get_id())

    def test_get_left_neuron(self):
        self.assertEqual(self.left_neuron, self.edge.get_left_neuron())

    def test_get_right_neuron(self):
        self.assertEqual(self.right_neuron, self.edge.get_right_neuron())

    def test_str(self):
        self.assertEqual("Edge between Neuron (4, 3) and Neuron (5, 2)",
                         str(self.edge))


if __name__ == '__main__':
    unittest.main()
