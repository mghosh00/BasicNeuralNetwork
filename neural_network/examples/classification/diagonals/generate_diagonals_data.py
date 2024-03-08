from neural_network import NormalDataGenerator


def classifier(x_1: float, x_2: float) -> str:
    """This classifies points according to diagonals x_1 = x_2 and x_1 = -x_2.

    Parameters
    ----------
    x_1 : float
        Horizontal coordinate
    x_2 : float
        Vertical coordinate

    Returns
    -------
    str
        Positional string (north, south, east, west)
    """
    if x_1 + x_2 > 0:
        if x_1 - x_2 > 0:
            return "East"
        else:
            return "North"
    else:
        if x_1 - x_2 > 0:
            return "South"
        else:
            return "West"


# Generate the data (using a multivariate normal) and save it to a .csv file
generator = NormalDataGenerator(function=classifier, num_datapoints=400,
                                means=[0.0, 0.0], std_devs=[1.0, 1.0])

data = generator()
generator.write_to_csv("diagonals_data")
