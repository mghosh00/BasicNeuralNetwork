import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network import Network

from neural_network import Tester


class TestTester(TestCase):
    """Tests the `Tester` class
    """
    partitions = [[2, 4, 0], [3, 1]]
    weighted_partitions = [[1, 3, 2], [0, 4, 3]]
    batch_losses = [0.5, 0.2]

    def setUp(self):
        self.network = Network(num_features=3, num_hidden_layers=2,
                               neuron_counts=[4, 3])
        self.test_data = np.array([[-2, 0, 3, 1],
                                   [2, 6, -9, 1],
                                   [-8, -2, 9, 0],
                                   [-8, 4, 1, 0],
                                   [2, 6, -8, 0]])
        self.df = pd.DataFrame(self.test_data,
                               columns=["a", "b", "c", "class"])
        self.default_tester = Tester(self.network, self.df, batch_size=3)
        self.tester = Tester(self.network, self.df, batch_size=3,
                             weighted=True, classification=False)

    def test_construct(self):
        self.assertEqual(self.network, self.default_tester._network)
        self.assertFalse(self.tester._classification)

    @mock.patch('neural_network.main.tester.Tester.forward_pass_one_batch',
                side_effect=batch_losses)
    @mock.patch('neural_network.util.partitioner.Partitioner.__call__')
    @mock.patch('builtins.print')
    def test_run_default(self, mock_print, mock_partition,
                         mock_forward_pass):
        # Here we use the different static lists of the class to dictate what
        # each of the above functions returns
        mock_partition.return_value = self.partitions

        # This is from averaging the testing losses over 5 data points
        expected_loss = 0.14
        self.default_tester.run()

        self.assertEqual(1, mock_partition.call_count)
        partition_calls = [mock.call(self.partitions[i]) for i in range(2)]
        mock_forward_pass.assert_has_calls(partition_calls)

        # 2 calls only
        self.assertEqual(2, mock_forward_pass.call_count)

        print_calls = [mock.call(f"Testing loss: {expected_loss}")]
        mock_print.assert_has_calls(print_calls)
        self.assertEqual(1, mock_print.call_count)

    @mock.patch('neural_network.main.tester.Tester'
                '.forward_pass_one_batch',
                side_effect=batch_losses)
    @mock.patch('neural_network.util.weighted_partitioner'
                '.WeightedPartitioner.__call__')
    def test_run(self, mock_partition, mock_forward_pass):
        # Here we use the different static lists of the class to dictate what
        # each of the above functions returns
        mock_partition.return_value = self.weighted_partitions

        self.tester.run()

        self.assertEqual(1, mock_partition.call_count)
        partition_calls = [mock.call(self.weighted_partitions[i])
                           for i in range(2)]
        mock_forward_pass.assert_has_calls(partition_calls)

    @mock.patch('neural_network.main.abstract_simulator.AbstractSimulator'
                '.abs_generate_scatter')
    def test_generate_scatter(self, mock_generator):
        self.tester.generate_scatter('test_title')
        mock_generator.assert_called_once_with(phase='testing',
                                               title='test_title')

    @mock.patch('builtins.print')
    def test_generate_confusion(self, mock_print):
        self.tester._data['y_hat'] = np.array([1, 0, 0, 0, 0])
        self.tester.generate_confusion()
        expected_confusion_array = np.array([[3, 0], [1, 1]],
                                            dtype='int64')
        expected_dice_scores = {0: 6 / 7, 1: 2 / 3}
        expected_average_dice_score = 16 / 21
        print_calls = [mock.call("Confusion matrix for testing data:"),
                       mock.call(expected_confusion_array),
                       mock.call(f"Dice scores: {expected_dice_scores}"),
                       mock.call(f"Mean dice score: "
                                 f"{expected_average_dice_score}")]
        call_list = mock_print.call_args_list
        for i in range(4):
            if i == 1:
                np.testing.assert_array_equal(print_calls[i].args[0],
                                              call_list[i].args[0].to_numpy())
            else:
                self.assertEqual(print_calls[i], call_list[i])
        self.assertEqual(4, mock_print.call_count)


if __name__ == '__main__':
    unittest.main()
