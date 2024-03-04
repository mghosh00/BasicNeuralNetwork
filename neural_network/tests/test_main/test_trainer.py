import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network import Network

from neural_network import Validator
from neural_network import Trainer


class TestTrainer(TestCase):
    """Tests the `Trainer` class
    """
    partitions = [[6, 2], [1, 8], [0, 3], [4, 7], [9, 5]]
    weighted_partitions = [[5, 2], [2, 8], [4, 1], [8, 6], [9, 3]]
    batch_losses = [0.1, 0.2, 0.3, 0.2, 0.5,
                    0.6, 0.2, 0.3, 0.1, 0.3,
                    0.2, 0.1, 0.0, 0.0, 0.2,
                    0.1, 0.1, 0.1, 0.1, 0.1,
                    0.0, 0.1, 0.0, 0.1, 0.0]
    validation_losses = [0.9, 0.7, 0.5, 0.3, 0.1]

    def setUp(self):
        self.network = Network(num_features=3, num_hidden_layers=2,
                               neuron_counts=[4, 3])
        self.validation_data = np.array([[4, 1, 3, 1],
                                         [2, 5, -4, 1],
                                         [-2, -4, 1, 0],
                                         [-9, 2, 4, 0]])
        self.validation_df = pd.DataFrame(self.validation_data,
                                          columns=["a", "b", "c", "class"])
        self.validator = Validator(self.network, self.validation_df,
                                   batch_size=1)
        self.train_data = np.array([[3, 2, 5, 1],
                                    [6, -2, -3, 1],
                                    [0, 1, 0, 1],
                                    [-4, -3, -2, 0],
                                    [1, -9, 2, 0],
                                    [2, 4, -3, 1],
                                    [-4, -2, 5, 0],
                                    [2, 3, 1, 1],
                                    [-9, -3, 2, 0],
                                    [2, 3, -4, 1]])
        self.df = pd.DataFrame(self.train_data,
                               columns=["a", "b", "c", "class"])
        self.default_trainer = Trainer(self.network, self.df, num_epochs=5,
                                       batch_size=2)
        self.trainer = Trainer(self.network, self.df, num_epochs=5,
                               batch_size=2, validator=self.validator,
                               weighted=True, classification=False)

    def test_construct_default(self):
        self.assertEqual(self.network, self.default_trainer._network)
        self.assertEqual(5, self.default_trainer._num_epochs)
        self.assertIsNone(self.default_trainer._validator)
        pd.testing.assert_frame_equal(pd.DataFrame(columns=['Training']),
                                      self.default_trainer._loss_df)

    def test_construct_non_default(self):
        self.assertEqual(5, self.trainer._num_epochs)
        self.assertEqual(self.validator, self.trainer._validator)
        pd.testing.assert_frame_equal(pd.DataFrame(columns=['Training',
                                                            'Validation']),
                                      self.trainer._loss_df)

    @mock.patch('neural_network.components.network.Network'
                '.store_gradient_of_loss')
    def test_store_gradients(self, mock_store):
        _id = 4
        self.trainer.store_gradients(_id)
        edges = self.network.get_edges()
        calls = []
        for layer in reversed(edges):
            for right_neuron in layer:
                for edge in right_neuron:
                    # If it is the first left neuron in the layer, we call
                    # the store with first=True
                    if edge.get_id()[1] == 0:
                        calls.append(mock.call(edge, 0, True))
                    # Else we call it with first=False
                    else:
                        calls.append(mock.call(edge, 0, False))

        # We have 3 * 4 + 4 * 3 + 3 * 2 edges in total
        self.assertEqual(30, mock_store.call_count)
        mock_store.assert_has_calls(calls)

    @mock.patch('neural_network.components.network.Network'
                '.back_propagate_bias')
    @mock.patch('neural_network.components.network.Network'
                '.back_propagate_weight')
    def test_back_propagate_one_batch(self, mock_weight_propagator,
                                      mock_bias_propagator):
        self.trainer.back_propagate_one_batch()

        edges = self.network.get_edges()
        weight_calls = []
        for layer in reversed(edges):
            for right_neuron in layer:
                for edge in right_neuron:
                    weight_calls.append(mock.call(edge))
        self.assertEqual(30, mock_weight_propagator.call_count)
        mock_weight_propagator.assert_has_calls(weight_calls)

        layers = self.network.get_layers()
        bias_calls = []
        for layer in layers[1:]:
            for neuron in layer.get_neurons():
                bias_calls.append(mock.call(neuron))
        # We have 4 + 3 + 2 neurons not including the input layer
        self.assertEqual(9, mock_bias_propagator.call_count)
        mock_bias_propagator.assert_has_calls(bias_calls)

    @mock.patch('neural_network.main.trainer.Trainer'
                '._update_categorical_dataframe')
    @mock.patch('neural_network.main.trainer.Trainer'
                '.back_propagate_one_batch')
    @mock.patch('neural_network.main.trainer.Trainer.forward_pass_one_batch',
                side_effect=batch_losses)
    @mock.patch('neural_network.util.partitioner.Partitioner.__call__')
    @mock.patch('builtins.print')
    def test_run_default(self, mock_print, mock_partition, mock_forward_pass,
                         mock_back_propagate, mock_update_frame):
        # Here we use the different static lists of the class to dictate what
        # each of the above functions returns
        mock_partition.return_value = self.partitions
        self.default_trainer.run()
        self.assertEqual(5, mock_partition.call_count)
        partition_calls = [mock.call(self.partitions[i]) for i in range(5)]
        mock_forward_pass.assert_has_calls(partition_calls)

        # 5 epochs * 5 batches per epoch
        self.assertEqual(25, mock_forward_pass.call_count)
        self.assertEqual(25, mock_back_propagate.call_count)

        # These are calculated by averaging each set of 5 batch_losses and then
        # dividing by 2 (as batch_size = 2)
        losses = [0.13, 0.15, 0.05, 0.05, 0.02]
        print_calls = []
        for i in range(5):
            print_calls.append(mock.call(f"Epoch: {i}"))
            print_calls.append(mock.call(f"Loss: {losses[i]}"))
        mock_print.assert_has_calls(print_calls)
        self.assertEqual(10, mock_print.call_count)

        # Finally check that the loss dataframe is as expected
        expected_df = pd.DataFrame({'Training': np.array(losses,
                                                         dtype=object)})
        pd.testing.assert_frame_equal(expected_df,
                                      self.default_trainer._loss_df)
        mock_update_frame.assert_called_once()

    @mock.patch('neural_network.main.trainer.Trainer'
                '._update_categorical_dataframe')
    @mock.patch('neural_network.main.validator.Validator.validate',
                side_effect=validation_losses)
    @mock.patch('neural_network.main.trainer.Trainer'
                '.back_propagate_one_batch')
    @mock.patch('neural_network.main.trainer.Trainer.forward_pass_one_batch',
                side_effect=batch_losses)
    @mock.patch('neural_network.util.weighted_partitioner'
                '.WeightedPartitioner.__call__')
    @mock.patch('builtins.print')
    def test_run(self, mock_print, mock_partition, mock_forward_pass,
                 mock_back_propagate, mock_validate, mock_update_frame):
        # Here we use the different static lists of the class to dictate what
        # each of the above functions returns
        mock_partition.return_value = self.weighted_partitions
        self.trainer.run()
        self.assertEqual(5, mock_partition.call_count)
        partition_calls = [mock.call(self.weighted_partitions[i])
                           for i in range(5)]
        mock_forward_pass.assert_has_calls(partition_calls)

        # These are calculated by averaging each set of 5 batch_losses and then
        # dividing by 2 (as batch_size = 2)
        losses = [0.13, 0.15, 0.05, 0.05, 0.02]
        print_calls = []
        for i in range(5):
            print_calls.append(mock.call(f"Epoch: {i}"))
            print_calls.append(mock.call(f"Loss: {losses[i]}"))
        mock_print.assert_has_calls(print_calls)
        self.assertEqual(10, mock_print.call_count)

        # Check validation occurs
        mock_validate.assert_has_calls([mock.call(1)] * 5)
        self.assertEqual(5, mock_validate.call_count)

        # Finally check that the loss dataframe is as expected
        expected_df = pd.DataFrame({'Training': np.array(losses,
                                                         dtype=object),
                                    'Validation':
                                        np.array(self.validation_losses,
                                                 dtype=object)})
        pd.testing.assert_frame_equal(expected_df,
                                      self.trainer._loss_df)
        mock_update_frame.assert_called_once()

    @mock.patch('neural_network.main.abstract_simulator.AbstractSimulator'
                '.abs_generate_scatter')
    def test_generate_scatter(self, mock_generator):
        self.trainer.generate_scatter('test_title')
        mock_generator.assert_called_once_with(phase='training',
                                               title='test_title')

    @mock.patch('neural_network.main.plotter.Plotter.plot_loss')
    def test_generate_loss_plot(self, mock_plotter):
        self.trainer.generate_loss_plot('test_title')
        mock_plotter.assert_called_once_with(self.trainer._loss_df,
                                             'test_title')


if __name__ == '__main__':
    unittest.main()
