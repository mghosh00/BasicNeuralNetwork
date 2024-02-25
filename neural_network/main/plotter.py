import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, ggtitle
from plotnine.animation import PlotnineAnimation


class Plotter:
    """Class to track progress of training/testing
    """

    path = "plots/"

    def __init__(self):
        self._frames = []

    def plot_predictions(self, df: pd.DataFrame, title: str = ''):
        """Creates a scatter plot of the predicted classes for a given set
        of data

        Parameters
        ----------
        df : pd.DataFrame
            The data
        title : str
            The title
        """
        plot = (ggplot(df, aes(x='x_1', y='x_2'))
                + geom_point(aes(color='factor(y_hat)'))
                + ggtitle("Predicted classes for data")
                )
        self._frames.append(plot)
        substring = '_' + title if title else ''
        plot.save(Plotter.path + f"scatter{substring}.png")

    @staticmethod
    def plot_loss(df: pd.DataFrame, title: str = ''):
        """Plots (normally) the training and validation losses over time.

        Parameters
        df : pd.DataFrame
            The data
        """
        times = np.linspace(0, len(df), len(df))
        for column in df.columns:
            plt.plot(times, df[column].to_numpy(), label=column)
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy loss")
        plt.legend()
        plt.title("Loss over time")
        substring = '_' + title if title else ''
        plt.savefig(Plotter.path + f"losses{substring}.png")

    def plot_predictions_gif(self):
        """Produces gif of scatter plots
        """
        plots = (frame for frame in self._frames)
        animation = PlotnineAnimation(plots, interval=100, repeat_delay=500)
        animation.save("gifs/scatter.mp4")
