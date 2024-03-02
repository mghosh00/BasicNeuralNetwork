import sys
import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network import Plotter


class TestPlotter(TestCase):
    """Tests the `Plotter` class
    """
    partitions = [[2, 4, 0], [3, 1]]
    weighted_partitions = [[1, 3, 2], [0, 4, 3]]
    batch_losses = [0.5, 0.2]
    ggplot_string = ("plotnine.ggplot.ggplot.save"
                     if sys.version_info[1] > 10 else "builtins.print")

    def setUp(self):
        self.scatter_data = np.array([[-2, 0, 1, 1],
                                      [2, 6, 1, 0],
                                      [-8, -2, 0, 0],
                                      [-8, 4, 0, 0],
                                      [2, 6, 0, 1]])
        self.scatter_df = pd.DataFrame(self.scatter_data,
                                       columns=["x_1", "x_2", "y", "y_hat"])
        self.loss_data = np.array([[0.8, 0.9],
                                   [0.7, 0.8],
                                   [0.6, 0.7],
                                   [0.5, 0.6],
                                   [0.4, 0.5],
                                   [0.3, 0.4],
                                   [0.2, 0.3],
                                   [0.1, 0.2]])
        self.loss_df = pd.DataFrame(self.loss_data,
                                    columns=['Training', 'Validation'])

    def test_construct(self):
        self.assertEqual(Plotter.path, "plots/")

    @mock.patch(ggplot_string)
    def test_plot_predictions_default(self, mock_save):
        if sys.version_info[1] > 10:
            Plotter.plot_predictions(self.scatter_df)
            mock_save.assert_called_with("plots/training/scatter.png")
        else:
            assert True

    @mock.patch(ggplot_string)
    def test_plot_predictions(self, mock_save):
        if sys.version_info[1] > 10:
            Plotter.plot_predictions(self.scatter_df, 'validation',
                                     'test_title')
            mock_save.assert_called_with("plots/validation/"
                                         "scatter_test_title.png")
        else:
            assert True

    @mock.patch('matplotlib.pyplot.savefig')
    @mock.patch('matplotlib.pyplot.title')
    @mock.patch('matplotlib.pyplot.legend')
    @mock.patch('matplotlib.pyplot.ylabel')
    @mock.patch('matplotlib.pyplot.xlabel')
    @mock.patch('matplotlib.pyplot.plot')
    def test_plot_loss_default(self, mock_plt, mock_xlabel, mock_ylabel,
                               mock_legend, mock_title, mock_save):
        Plotter.plot_loss(self.loss_df)
        plot_call_list = mock_plt.call_args_list
        first_args = plot_call_list[0].args
        second_args = plot_call_list[1].args
        times = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(first_args[0], times)
        np.testing.assert_array_equal(second_args[0], times)
        np.testing.assert_array_equal(first_args[1], self.loss_data[:, 0])
        np.testing.assert_array_equal(second_args[1], self.loss_data[:, 1])
        self.assertDictEqual(plot_call_list[0].kwargs,
                             {'label': 'Training'})
        self.assertDictEqual(plot_call_list[1].kwargs,
                             {'label': 'Validation'})
        mock_xlabel.assert_called_once_with("Epoch")
        mock_ylabel.assert_called_once_with("Cross entropy loss")
        mock_legend.assert_called_once()
        mock_title.assert_called_once_with("Loss over time")
        mock_save.assert_called_once_with("plots/losses.png")

    @mock.patch('matplotlib.pyplot.savefig')
    @mock.patch('matplotlib.pyplot.plot')
    def test_plot_loss(self, mock_plt, mock_save):
        Plotter.plot_loss(self.loss_df, "test_title")
        mock_save.assert_called_once_with("plots/losses_test_title.png")


if __name__ == '__main__':
    unittest.main()
