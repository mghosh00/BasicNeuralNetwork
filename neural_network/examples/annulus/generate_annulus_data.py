from neural_network import NormalDataGenerator


def classifier(x_1: float, x_2: float) -> str:
    """This classifies points according to whether they lie inside a disc,
    annulus or outside them both.

    Parameters
    ----------
    x_1 : float
        Horizontal coordinate
    x_2 : float
        Vertical coordinate

    Returns
    -------
    str
        Whether the point lies in "Disc", "Annulus" or "Outside" both regions
    """
    if x_1**2 + x_2**2 < 1:
        return "Disc"
    elif 1 <= x_1**2 + x_2**2 < 4:
        return "Annulus"
    else:
        return "Outside"


# Generate the data (using a uniform distribution) and save it to a .csv file
generator = NormalDataGenerator(classifier=classifier, num_datapoints=1000,
                                means=[0.0, 0.0],
                                std_devs=[1.5, 1.5])

data = generator()
generator.write_to_csv("annulus_data")
