from neural_network import NormalDataGenerator


def mean_regressor(x_1: float, x_2: float) -> float:
    """This simply returns the mean of the two coordinates.

    Parameters
    ----------
    x_1 : float
        Horizontal coordinate
    x_2 : float
        Vertical coordinate

    Returns
    -------
    float
        The mean of the two coordinates
    """
    return (x_1 + x_2) / 2


# Generate the data (using a normal distribution) and save it to a .csv file
generator = NormalDataGenerator(function=mean_regressor, num_datapoints=800,
                                means=[0.0, 0.0], std_devs=[5.0, 5.0])

data = generator()
generator.write_to_csv("mean_data")
