import sys
import unittest
from unittest import TestCase
from unittest import mock

import numpy as np
import pandas as pd

from neural_network.learning import Plotter


class TestPlotter(TestCase):
    """Tests the `Plotter` class
    """
    ggplot_string = ("plotnine.ggplot.ggplot.save"
                     if sys.version_info[1] > 10 else "builtins.len")

    def setUp(self):
        Plotter.show_plots = False
        self.scatter_data = np.array([[-2, 0, 1, 1],
                                      [2, 6, 1, 0],
                                      [-8, -2, 0, 0],
                                      [-8, 4, 0, 0],
                                      [2, 6, 0, 1]])
        self.scatter_df = pd.DataFrame(self.scatter_data,
                                       columns=["x_1", "x_2", "y", "y_hat"])
        self.reg_scatter_data = np.array([[-2, 0, 0.0, 0.3],
                                          [2, 6, 6.0, 5.7],
                                          [-8, -2, -2.0, -4.2],
                                          [-8, 4, 4.0, 4.1],
                                          [2, 6, 6.0, 5.5]])
        self.reg_scatter_df = pd.DataFrame(self.reg_scatter_data,
                                           columns=["x_1", "x_2", "y",
                                                    "y_hat"])
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
    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    @mock.patch('builtins.print')
    def test_datapoint_scatter_default(self, mock_print, mock_exists,
                                       mock_makedirs, mock_save):
        if sys.version_info[1] > 10:
            mock_exists.return_value = False
            Plotter.datapoint_scatter(self.scatter_df)
            mock_print.assert_not_called()
            exists_calls = mock_exists.call_args_list
            self.assertListEqual([mock.call("plots/"),
                                  mock.call("plots/training")], exists_calls)
            self.assertListEqual(exists_calls, mock_makedirs.call_args_list)
            self.assertEqual(mock_exists.call_count, 2)
            self.assertEqual(mock_makedirs.call_count, 2)
            mock_save.assert_called_once_with("plots/training/scatter.png")
        else:
            self.skipTest("plotnine testing incompatible with python 3.10")

    @mock.patch(ggplot_string)
    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    @mock.patch('builtins.print')
    def test_datapoint_scatter(self, mock_print, mock_exists, mock_makedirs,
                               mock_save):
        if sys.version_info[1] > 10:
            mock_exists.return_value = True
            Plotter.datapoint_scatter(self.scatter_df, 'validation',
                                      'test_title')
            mock_print.assert_not_called()
            exists_calls = mock_exists.call_args_list
            self.assertListEqual([mock.call("plots/"),
                                  mock.call("plots/validation")], exists_calls)
            self.assertEqual(mock_exists.call_count, 2)
            self.assertEqual(mock_makedirs.call_count, 0)
            mock_save.assert_called_with("plots/validation/"
                                         "scatter_test_title.png")
        else:
            self.skipTest("plotnine testing incompatible with python 3.10")

    @mock.patch(ggplot_string)
    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    @mock.patch('builtins.print')
    def test_datapoint_scatter_regression(self, mock_print, mock_exists,
                                          mock_makedirs, mock_save):
        if sys.version_info[1] > 10:
            mock_exists.return_value = True
            Plotter.datapoint_scatter(self.scatter_df, 'validation',
                                      'test_title', regression=True)
            mock_print.assert_not_called()
            exists_calls = mock_exists.call_args_list
            self.assertListEqual([mock.call("plots/"),
                                  mock.call("plots/validation")], exists_calls)
            self.assertEqual(mock_exists.call_count, 2)
            self.assertEqual(mock_makedirs.call_count, 0)
            mock_save.assert_called_with("plots/validation/"
                                         "scatter_test_title.png")
        else:
            self.skipTest("plotnine testing incompatible with python 3.10")

    @mock.patch(ggplot_string)
    @mock.patch('builtins.print')
    def test_datapoint_scatter_true(self, mock_print, mock_save):
        if sys.version_info[1] > 10:
            Plotter.datapoint_scatter(self.scatter_df, 'true',
                                      'test_title', regression=True)
            mock_print.assert_not_called()
            mock_save.assert_called_with("true_scatter_test_title.png")
        else:
            self.skipTest("plotnine testing incompatible with python 3.10")

    @mock.patch(ggplot_string)
    @mock.patch('builtins.print')
    def test_datapoint_scatter_show(self, mock_print, mock_save):
        if sys.version_info[1] > 10:
            Plotter.show_plots = True
            Plotter.datapoint_scatter(self.scatter_df, 'true',
                                      'test_title', regression=True)
            mock_print.assert_called_once()
            mock_save.assert_called_with("true_scatter_test_title.png")
        else:
            self.skipTest("plotnine testing incompatible with python 3.10")

    @mock.patch(ggplot_string)
    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    @mock.patch('builtins.print')
    def test_comparison_scatter(self, mock_print, mock_exists, mock_makedirs,
                                mock_save):
        if sys.version_info[1] > 10:
            mock_exists.return_value = False
            Plotter.comparison_scatter(self.reg_scatter_df)
            mock_print.assert_not_called()
            exists_calls = mock_exists.call_args_list
            self.assertListEqual([mock.call("plots/"),
                                  mock.call("plots/training")], exists_calls)
            self.assertListEqual(exists_calls, mock_makedirs.call_args_list)
            self.assertEqual(mock_exists.call_count, 2)
            self.assertEqual(mock_makedirs.call_count, 2)
            mock_save.assert_called_once_with("plots/training/comparison.png")
        else:
            self.skipTest("plotnine testing incompatible with python 3.10")

    @mock.patch(ggplot_string)
    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    @mock.patch('builtins.print')
    def test_comparison_scatter_show(self, mock_print, mock_exists,
                                     mock_makedirs, mock_save):
        if sys.version_info[1] > 10:
            mock_exists.return_value = False
            Plotter.show_plots = True
            Plotter.comparison_scatter(self.reg_scatter_df)
            mock_print.assert_called_once()
            exists_calls = mock_exists.call_args_list
            self.assertListEqual([mock.call("plots/"),
                                  mock.call("plots/training")], exists_calls)
            self.assertListEqual(exists_calls, mock_makedirs.call_args_list)
            self.assertEqual(mock_exists.call_count, 2)
            self.assertEqual(mock_makedirs.call_count, 2)
            mock_save.assert_called_once_with("plots/training/comparison.png")
        else:
            self.skipTest("plotnine testing incompatible with python 3.10")

    @mock.patch('matplotlib.pyplot.savefig')
    @mock.patch('matplotlib.pyplot.title')
    @mock.patch('matplotlib.pyplot.legend')
    @mock.patch('matplotlib.pyplot.ylabel')
    @mock.patch('matplotlib.pyplot.xlabel')
    @mock.patch('matplotlib.pyplot.plot')
    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    def test_plot_loss_default(self, mock_exists, mock_makedirs, mock_plt,
                               mock_xlabel, mock_ylabel, mock_legend,
                               mock_title, mock_save):
        mock_exists.return_value = True
        Plotter.plot_loss(self.loss_df)
        mock_exists.assert_called_once_with("plots/")
        self.assertEqual(mock_makedirs.call_count, 0)
        plot_call_list = mock_plt.call_args_list
        first_args = plot_call_list[0].args
        second_args = plot_call_list[1].args
        times = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(first_args[0], times)
        np.testing.assert_array_equal(second_args[0], times)
        np.testing.assert_array_equal(first_args[1], self.loss_data[:, 1])
        np.testing.assert_array_equal(second_args[1], self.loss_data[:, 0])
        self.assertDictEqual(plot_call_list[0].kwargs,
                             {'label': 'Validation'})
        self.assertDictEqual(plot_call_list[1].kwargs,
                             {'label': 'Training'})
        mock_xlabel.assert_called_once_with("Epoch")
        mock_ylabel.assert_called_once_with("Loss")
        mock_legend.assert_called_once()
        mock_title.assert_called_once_with("Loss over time")
        mock_save.assert_called_once_with("plots/losses.png")

    @mock.patch('matplotlib.pyplot.savefig')
    @mock.patch('matplotlib.pyplot.plot')
    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    def test_plot_loss(self, mock_exists, mock_makedirs, mock_plt, mock_save):
        mock_exists.return_value = False
        Plotter.plot_loss(self.loss_df, "test_title")
        mock_exists.assert_called_with("plots/")
        mock_makedirs.assert_called_with("plots/")
        mock_save.assert_called_once_with("plots/losses_test_title.png")

    @mock.patch('matplotlib.pyplot.savefig')
    @mock.patch('matplotlib.pyplot.show')
    @mock.patch('matplotlib.pyplot.plot')
    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    def test_plot_loss_show(self, mock_exists, mock_makedirs, mock_plt,
                            mock_show, mock_save):
        mock_exists.return_value = False
        Plotter.show_plots = True
        Plotter.plot_loss(self.loss_df, "test_title")
        mock_exists.assert_called_with("plots/")
        mock_makedirs.assert_called_with("plots/")
        mock_show.assert_called_once()
        mock_save.assert_called_once_with("plots/losses_test_title.png")


if __name__ == '__main__':
    unittest.main()
