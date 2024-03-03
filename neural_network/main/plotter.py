import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, ggtitle


class Plotter:
    """Class to track progress of training/testing
    """

    path = "plots/"

    @staticmethod
    def plot_predictions(df: pd.DataFrame, phase: str = 'training',
                         title: str = ''):
        """Creates a scatter plot of the predicted classes for a given set
        of data

        Parameters
        ----------
        df : pd.DataFrame
            The data
        phase : str
            The phase of learning (training/validation/testing)
        title : str
            The title
        """
        plot = (ggplot(df, aes(x='x_1', y='x_2'))
                + geom_point(aes(color='factor(y_hat)'))
                + ggtitle(f"Predicted classes for {phase} data")
                )
        substring = '_' + title if title else ''
        plot.save(Plotter.path + f"{phase}/scatter{substring}.png")

    @staticmethod
    def plot_loss(df: pd.DataFrame, title: str = ''):
        """Plots (normally) the training and validation losses over time.

        Parameters
        df : pd.DataFrame
            The data
        """
        times = np.linspace(0, len(df), len(df), endpoint=False)
        for column in reversed(df.columns):
            plt.plot(times, df[column].to_numpy(), label=column)
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy loss")
        plt.legend()
        plt.title("Loss over time")
        substring = '_' + title if title else ''
        plt.savefig(Plotter.path + f"losses{substring}.png")
