import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network import Partitioner
from neural_network import WeightedPartitioner
from neural_network import CrossEntropyLoss
from neural_network import MSELoss
from neural_network import Network

from neural_network import AbstractLearner


class TestAbstractLearner(TestCase):
    """Tests the `AbstractLearner` class
    """

    def setUp(self):
        self.network = Network(num_features=3, num_hidden_layers=2,
                               neuron_counts=[4, 3])
        self.numpy_data = np.array([[3, 2, 5, 1],
                                    [6, -2, -3, 1],
                                    [0, 1, 0, 1],
                                    [-4, -3, -2, 0],
                                    [1, -9, 2, 0],
                                    [2, 4, -3, 1],
                                    [-4, -2, 5, 0],
                                    [2, 3, 1, 1],
                                    [-9, -3, 2, 0],
                                    [2, 3, -4, 1]])
        self.df = pd.DataFrame(self.numpy_data,
                               columns=["a", "b", "c", "class"])
        self.df["class"] = ["r", "r", "r", "l", "l", "r", "l", "r", "l", "r"]
        self.default_simulator = AbstractLearner(self.network, self.df,
                                                 batch_size=2)
        self.simulator = AbstractLearner(self.network, self.df, batch_size=2,
                                         weighted=True)
        self.regression_network = Network(num_features=3, num_hidden_layers=2,
                                          neuron_counts=[4, 3],
                                          regression=True)
        self.regression_data = np.array([[3, 2, 5, 5.0],
                                         [6, -2, -3, 6.0],
                                         [0, 1, 0, 1.0],
                                         [-4, -3, -2, -2.0],
                                         [1, -9, 2, 2.0],
                                         [2, 4, -3, 4.0],
                                         [-4, -2, 5, 5.0],
                                         [2, 3, 1, 3.0],
                                         [-9, -3, 2, 2.0],
                                         [2, 3, -4, 3.0]])
        self.regression_df = pd.DataFrame(self.regression_data,
                                          columns=["a", "b", "c", "actual"])
        self.regression_simulator = AbstractLearner(self.regression_network,
                                                    self.regression_df,
                                                    batch_size=2,
                                                    weighted=True,
                                                    bins=5)

    def test_construct_erroneous(self):
        # 1. Not enough columns
        array_1 = np.array([[2, 2, 0], [2, 2, 1], [2, 2, 0]])
        df_1 = pd.DataFrame(array_1)
        with self.assertRaises(ValueError) as ve_1:
            AbstractLearner(self.network, df_1, batch_size=1)
        self.assertEqual("Number of features must match number of initial "
                         "neurons (features = 2, initial neurons = 3)",
                         str(ve_1.exception))

        # 2. Too many output labels for the size of the network
        array_2 = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        df_2 = pd.DataFrame(array_2)
        with self.assertRaises(ValueError) as ve_2:
            AbstractLearner(self.network, df_2, batch_size=1)
        self.assertEqual("The number of output neurons in the network "
                         "(2) is less than the number of "
                         "classes in the dataframe (3)",
                         str(ve_2.exception))

        # 3. Batch size too big
        with self.assertRaises(ValueError) as ve_4:
            AbstractLearner(self.network, self.df, batch_size=11)
        self.assertEqual("Batch size must be smaller than number of "
                         "datapoints", str(ve_4.exception))

    def test_construct_default(self):
        self.assertEqual(self.network, self.default_simulator._network)
        self.assertFalse(self.default_simulator._regression)
        self.assertListEqual(["l", "r"],
                             self.default_simulator._category_names)
        transformed_df = self.df.copy()
        transformed_df.columns = ['x_1', 'x_2', 'x_3', 'y']

        # Check that before the name change we get the categorical data, and
        # after we get the numerical data
        transformed_df['y_hat'] = [0] * 10
        pd.testing.assert_frame_equal(transformed_df,
                                      self.default_simulator._categorical_data)
        transformed_df['y'] = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        pd.testing.assert_frame_equal(transformed_df,
                                      self.default_simulator._data)
        self.assertEqual(2, self.default_simulator._batch_size)
        self.assertIsInstance(self.default_simulator._cross_entropy_loss,
                              CrossEntropyLoss)
        self.assertFalse(hasattr(self.default_simulator, '_mse_loss'))
        self.assertIsInstance(self.default_simulator._partitioner, Partitioner)

        # _n = len(data), _m = batch_size
        self.assertEqual(10, self.default_simulator._partitioner._n)
        self.assertEqual(2, self.default_simulator._partitioner._m)

    def test_construct_non_default(self):
        self.assertIsInstance(self.simulator._partitioner, WeightedPartitioner)
        self.assertEqual(10, self.simulator._partitioner._n)
        self.assertEqual(2, self.simulator._partitioner._m)
        self.assertEqual(2, self.simulator._partitioner._num_classes)

        # Each class and a list of indices corresponding to datapoints which
        # belong to the class
        class_dict = {0: [3, 4, 6, 8], 1: [0, 1, 2, 5, 7, 9]}
        self.assertDictEqual(class_dict,
                             self.simulator._partitioner._class_dict)

    def test_construct_regression(self):
        self.assertEqual(self.regression_network,
                         self.regression_simulator._network)
        self.assertTrue(self.regression_simulator._regression)
        transformed_df = self.regression_df.copy()
        transformed_df.columns = ['x_1', 'x_2', 'x_3', 'y']
        transformed_df['y_hat'] = [0.0] * 10
        pd.testing.assert_frame_equal(transformed_df,
                                      self.regression_simulator._data)
        self.assertIsInstance(self.regression_simulator._mse_loss, MSELoss)
        self.assertFalse(hasattr(self.regression_simulator,
                                 '_cross_entropy_loss'))
        self.assertFalse(hasattr(self.regression_simulator,
                                 '_categorical_data'))

    @mock.patch('neural_network.learning.abstract_learner.AbstractLearner.'
                'store_gradients')
    @mock.patch('neural_network.functions.cross_entropy_loss.'
                'CrossEntropyLoss.__call__',
                side_effect=[0.2, 0.3])
    @mock.patch('neural_network.components.network.'
                'Network.forward_pass_one_datapoint',
                side_effect=[[0.2, 0.8], [0.7, 0.3]])
    def test_forward_pass_one_batch(self, mock_softmax, mock_loss, mock_store):
        batch_ids = [7, 8]
        total_loss = self.simulator.forward_pass_one_batch(batch_ids)
        self.assertEqual(0.5, total_loss)
        self.assertEqual(1, self.simulator._data.at[7, 'y_hat'])
        self.assertEqual(0, self.simulator._data.at[8, 'y_hat'])
        self.assertEqual(mock_softmax.call_count, 2)
        mock_loss.assert_any_call([0.2, 0.8], 1)
        mock_loss.assert_any_call([0.7, 0.3], 0)
        self.assertEqual(mock_loss.call_count, 2)
        mock_store.assert_any_call(7)
        mock_store.assert_any_call(8)
        self.assertEqual(mock_store.call_count, 2)

    @mock.patch('neural_network.learning.abstract_learner.AbstractLearner.'
                'store_gradients')
    @mock.patch('neural_network.functions.mse_loss.MSELoss.__call__',
                side_effect=[0.2, 0.3])
    @mock.patch('neural_network.components.network.'
                'Network.forward_pass_one_datapoint',
                side_effect=[[3.1], [1.8]])
    def test_forward_pass_one_batch_regression(self, mock_regressor, mock_loss,
                                               mock_store):
        batch_ids = [7, 8]
        total_loss = (self.regression_simulator
                      .forward_pass_one_batch(batch_ids))
        self.assertEqual(0.5, total_loss)
        self.assertEqual(3.1, self.regression_simulator
                         ._data.at[7, 'y_hat'])
        self.assertEqual(1.8, self.regression_simulator
                         ._data.at[8, 'y_hat'])
        self.assertEqual(mock_regressor.call_count, 2)
        mock_loss.assert_any_call(3.1, 3.0)
        mock_loss.assert_any_call(1.8, 2.0)
        self.assertEqual(mock_loss.call_count, 2)
        mock_store.assert_any_call(7)
        mock_store.assert_any_call(8)
        self.assertEqual(mock_store.call_count, 2)

    def test_run(self):
        with self.assertRaises(NotImplementedError) as ve:
            self.simulator.run()
        self.assertEqual("Cannot call from base class", str(ve.exception))

    def test_store_gradients(self):
        result = self.simulator.store_gradients(4)
        self.assertIsNone(result)

    def test_update_categorical_dataframe_erroneous(self):
        with self.assertRaises(RuntimeError) as re:
            self.regression_simulator._update_categorical_dataframe()
        self.assertEqual("Cannot call _update_categorical_dataframe"
                         "with a regressional network", str(re.exception))

    def test_update_categorical_dataframe(self):
        self.default_simulator._data['y_hat'] = np.array([0, 1, 0, 0, 1,
                                                          0, 1, 1, 0, 1])
        self.default_simulator._update_categorical_dataframe()
        expected_df = self.default_simulator._categorical_data.copy()
        expected_df['y_hat'] = ["l", "r", "l", "l", "r",
                                "l", "r", "r", "l", "r"]
        pd.testing.assert_frame_equal(expected_df,
                                      self.default_simulator._categorical_data)

    @mock.patch('neural_network.learning.plotter.Plotter.datapoint_scatter')
    def test_abs_generate_scatter(self, mock_plot):
        self.simulator.abs_generate_scatter()
        mock_plot.assert_called_once_with(self.simulator._categorical_data,
                                          'training', '')

    @mock.patch('neural_network.learning.plotter.Plotter.datapoint_scatter')
    def test_abs_generate_scatter_regression(self, mock_plot):
        self.regression_simulator.abs_generate_scatter()
        mock_plot.assert_called_once_with(self.regression_simulator._data,
                                          'training', '', regression=True)

    @mock.patch('neural_network.learning.plotter.Plotter.comparison_scatter')
    def test_abs_comparison_scatter(self, mock_plot):
        self.regression_simulator.abs_comparison_scatter()
        mock_plot.assert_called_once_with(self.regression_simulator._data,
                                          'training', '')
        with self.assertRaises(RuntimeError) as re:
            self.simulator.abs_comparison_scatter()
        self.assertEqual("Cannot call this method with categorical data",
                         str(re.exception))


if __name__ == '__main__':
    unittest.main()
