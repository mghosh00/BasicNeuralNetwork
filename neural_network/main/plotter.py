import pandas as pd
from plotnine import ggplot, aes, geom_point
from plotnine.animation import PlotnineAnimation


class Plotter:
    """Class to track progress of training/testing
    """

    path = "plots/"

    def __init__(self):
        self._frames = []

    def plot_predictions(self, df: pd.DataFrame):
        plot = (ggplot(df, aes(x='x_1', y='x_2'))
                + geom_point(aes(color='factor(y_hat)'))
                )
        self._frames.append(plot)
        plot.save(Plotter.path + "scatter.png")

    def plot_predictions_gif(self):
        plots = (frame for frame in self._frames)
        animation = PlotnineAnimation(plots, interval=100, repeat_delay=500)
        animation.save("gifs/scatter.mp4")
