import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network import Network

from neural_network import Validator


class TestValidator(TestCase):
    """Tests the `Validator` class
    """
    partitions = [[2], [0], [1], [3]]
    weighted_partitions = [[2], [2], [1], [0]]
    batch_losses = [0.4, 0.2, 0.5, 0.1]

    def setUp(self):
        self.network = Network(num_features=3, num_hidden_layers=2,
                               neuron_counts=[4, 3])
        self.validation_data = np.array([[4, 1, 3, 1],
                                         [2, 5, -4, 1],
                                         [-2, -4, 1, 0],
                                         [-9, 2, 4, 0]])
        self.df = pd.DataFrame(self.validation_data,
                               columns=["a", "b", "c", "class"])
        self.default_validator = Validator(self.network, self.df, batch_size=1)
        self.validator = Validator(self.network, self.df, batch_size=1,
                                   weighted=True)
        self.regression_network = Network(num_features=3, num_hidden_layers=2,
                                          neuron_counts=[4, 3],
                                          regression=True)
        self.reg_validation_data = np.array([[4, 1, 3, 1.3],
                                             [2, 5, -4, 1.4],
                                             [-2, -4, 1, 0.1],
                                             [-9, 2, 4, 0.2]])
        self.reg_validation_df = pd.DataFrame(self.reg_validation_data,
                                              columns=["a", "b", "c",
                                                       "predicted"])
        self.reg_validator = Validator(self.regression_network,
                                       self.reg_validation_df, batch_size=1)

    def test_construct(self):
        self.assertEqual(self.network, self.default_validator._network)
        self.assertEqual(0, self.default_validator._epoch)
        self.assertFalse(self.validator._regression)
        self.assertEqual(0, self.validator._epoch)

    def test_construct_regression(self):
        self.assertTrue(self.reg_validator._regression)

    @mock.patch('neural_network.learning.validator.Validator'
                '._update_categorical_dataframe')
    @mock.patch('neural_network.learning.validator.Validator'
                '.forward_pass_one_batch',
                side_effect=batch_losses)
    @mock.patch('neural_network.util.partitioner.Partitioner.__call__')
    @mock.patch('builtins.print')
    def test_validate_default(self, mock_print, mock_partition,
                              mock_forward_pass, mock_update_frame):
        # Here we use the different static lists of the class to dictate what
        # each of the above functions returns
        mock_partition.return_value = self.partitions

        # This is from averaging the validation losses
        expected_loss = 0.3
        validation_loss = self.default_validator.validate(factor=1)
        self.assertEqual(validation_loss, expected_loss)

        self.assertEqual(1, mock_partition.call_count)
        partition_calls = [mock.call(self.partitions[i]) for i in range(4)]
        mock_forward_pass.assert_has_calls(partition_calls)

        # 4 calls only
        self.assertEqual(4, mock_forward_pass.call_count)

        print_calls = [mock.call(f"Validation loss: {expected_loss}")]
        mock_print.assert_has_calls(print_calls)
        self.assertEqual(1, mock_print.call_count)
        mock_update_frame.assert_called_once()

    @mock.patch('neural_network.learning.validator.Validator'
                '._update_categorical_dataframe')
    @mock.patch('neural_network.learning.validator.Validator'
                '.forward_pass_one_batch',
                side_effect=batch_losses)
    @mock.patch('neural_network.util.weighted_partitioner'
                '.WeightedPartitioner.__call__')
    def test_validate(self, mock_partition, mock_forward_pass,
                      mock_update_frame):
        # Here we use the different static lists of the class to dictate what
        # each of the above functions returns
        mock_partition.return_value = self.weighted_partitions

        # This is from averaging the validation losses
        expected_loss = 0.3
        validation_loss = self.validator.validate(factor=1)
        self.assertEqual(validation_loss, expected_loss)

        self.assertEqual(1, mock_partition.call_count)
        partition_calls = [mock.call(self.weighted_partitions[i])
                           for i in range(4)]
        mock_forward_pass.assert_has_calls(partition_calls)
        mock_update_frame.assert_called_once()
        self.assertEqual(1, self.validator._epoch)

    @mock.patch('neural_network.learning.validator.Validator'
                '._update_categorical_dataframe')
    def test_validate_regression(self, mock_update_frame):
        self.reg_validator.validate(1)
        self.assertEqual(mock_update_frame.call_count, 0)

    @mock.patch('neural_network.learning.abstract_learner.AbstractLearner.'
                'abs_generate_scatter')
    def test_generate_scatter(self, mock_generator):
        self.validator.generate_scatter('test_title')
        mock_generator.assert_called_once_with(phase='validation',
                                               title='test_title')

    @mock.patch('neural_network.learning.abstract_learner.AbstractLearner.'
                'abs_comparison_scatter')
    def test_comparison_scatter(self, mock_comparison):
        self.reg_validator.comparison_scatter('test_title')
        mock_comparison.assert_called_once_with(phase='validation',
                                                title='test_title')


if __name__ == '__main__':
    unittest.main()
