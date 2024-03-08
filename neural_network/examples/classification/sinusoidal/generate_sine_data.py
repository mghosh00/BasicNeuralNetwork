import math

from neural_network import UniformDataGenerator


def classifier(x_1: float, x_2: float) -> str:
    """This classifies points according to whether the 2d point lies above or
    below the curve x_2 = sin(x_1).

    Parameters
    ----------
    x_1 : float
        Horizontal coordinate
    x_2 : float
        Vertical coordinate

    Returns
    -------
    str
        Whether the point lies "Above" or "Below" the curve x_2 = sin(x_1)
    """
    if x_2 > math.sin(x_1):
        return "Above"
    else:
        return "Below"


# Generate the data (using a uniform distribution) and save it to a .csv file
generator = UniformDataGenerator(function=classifier, num_datapoints=800,
                                 lower_bounds=[-6.28, -2.0],
                                 upper_bounds=[6.28, 2.0])

data = generator()
generator.write_to_csv("sine_data")
