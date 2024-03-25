import math

from neural_network import UniformDataGenerator


def sine_regressor(x_1: float, x_2: float) -> float:
    """This returns the product of sines.

    Parameters
    ----------
    x_1 : float
        Horizontal coordinate
    x_2 : float
        Vertical coordinate

    Returns
    -------
    float
        The product of sines
    """
    return math.sin(x_1) * math.sin(x_2)


# Generate the data (using a uniform distribution) and save it to a .csv file
generator = UniformDataGenerator(function=sine_regressor, num_datapoints=800,
                                 upper_bounds=[3.14, 3.14],
                                 lower_bounds=[-3.14, -3.14])

data = generator()
generator.write_to_csv("sine_data")
