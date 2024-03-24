import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, ggtitle, geom_abline


class Plotter:
    """Class to track progress of training/testing
    """

    path = "plots/"

    @staticmethod
    def datapoint_scatter(df: pd.DataFrame, phase: str = 'training',
                          title: str = '', regression: bool = False):
        """Creates a scatter plot of the predicted/true classes for a given set
        of data.

        Parameters
        ----------
        df : pd.DataFrame
            The data
        phase : str
            The phase of learning (training/validation/testing) or true for
            just the true data
        title : str
            The title
        regression : bool
            Whether this is for regressional data or classificational data
        """
        if not os.path.exists(Plotter.path):
            os.makedirs(Plotter.path)

        y_variable = 'y' if phase == 'true' else 'y_hat'
        actual_or_predicted = 'Actual' if phase == 'true' else 'Predicted'
        colouring_string = (y_variable if regression
                            else f'factor({y_variable})')
        values_or_classes = 'values' if regression else 'classes'

        plot = (ggplot(df, aes(x='x_1', y='x_2'))
                + geom_point(aes(color=colouring_string))
                + ggtitle(f"{actual_or_predicted} {values_or_classes} "
                          f"for {phase} data")
                )
        substring = '_' + title if title else ''

        if phase == 'true':
            plot.save(f"true_scatter{substring}.png")
        else:
            if not os.path.exists(Plotter.path + phase):
                os.makedirs(Plotter.path + phase)
            plot.save(Plotter.path + f"{phase}/scatter{substring}.png")

    @staticmethod
    def comparison_scatter(df: pd.DataFrame, phase: str = 'training',
                           title: str = ''):
        """Creates a scatter plot comparing the true and predicted values from
        the network.

        Parameters
        ----------
        df : pd.DataFrame
            The data
        phase : str
            The phase of learning (training/validation/testing)
        title : str
            The title
        """
        if not os.path.exists(Plotter.path):
            os.makedirs(Plotter.path)
        df = df.rename(columns={'y': 'Actual', 'y_hat': 'Predicted'})
        plot = (ggplot(df, aes(x='Actual', y='Predicted'))
                + geom_point()
                + geom_abline(colour='red')
                + ggtitle(f"Comparison scatter plot for {phase} data")
                )
        substring = '_' + title if title else ''
        if not os.path.exists(Plotter.path + phase):
            os.makedirs(Plotter.path + phase)
        plot.save(Plotter.path + f"{phase}/comparison{substring}.png")

    @staticmethod
    def plot_loss(df: pd.DataFrame, title: str = ''):
        """Plots (normally) the training and validation losses over time.

        Parameters
        df : pd.DataFrame
            The data
        """
        if not os.path.exists(Plotter.path):
            os.makedirs(Plotter.path)

        times = np.linspace(0, len(df), len(df), endpoint=False)
        for column in reversed(df.columns):
            plt.plot(times, df[column].to_numpy(), label=column)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss over time")
        substring = '_' + title if title else ''
        plt.savefig(Plotter.path + f"losses{substring}.png")
