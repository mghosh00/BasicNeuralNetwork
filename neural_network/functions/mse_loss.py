class MSELoss:
    """Class to represent the mean squared error loss for regressional neural
    networks.
    """

    def __init__(self):
        """Constructor method
        """
        pass

    def __call__(self, y_hat: float, y: float) -> float:
        """The loss function.

        Parameters
        ----------
        y_hat : float
            Output value from neuron in output layer
        y : float
            Ground truth value

        Returns
        -------
        float
            Squared difference of the two values
        """
        return round((y_hat - y) ** 2, 8)

    def gradient(self, y_hat: float, y: float) -> float:
        """The gradient of the loss function.

        Parameters
        ----------
        y_hat : float
            Output value from neuron in output layer
        y : float
            Ground truth value

        Returns
        -------
        float
            Difference of the two values multiplied by 2
        """
        return round(2 * (y_hat - y), 8)
