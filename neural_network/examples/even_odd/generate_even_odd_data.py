from neural_network import UniformDataGenerator


def classifier(x_1: float, x_2: float) -> str:
    """This classifies points according to whether the integer part of x_1
    is even or odd.

    Parameters
    ----------
    x_1 : float
        Horizontal coordinate
    x_2 : float
        Vertical coordinate

    Returns
    -------
    str
        Whether integer part of x_1 is Even or Odd
    """
    int_x_1 = int(x_1)
    if int_x_1 % 2 == 0:
        return "Even"
    else:
        return "Odd"


# Generate the data (using a uniform distribution) and save it to a .csv file
generator = UniformDataGenerator(classifier=classifier, num_datapoints=400,
                                 lower_bounds=[0.0, 0.0],
                                 upper_bounds=[6.0, 6.0])

data, categories = generator()
generator.write_to_csv("even_odd_data")
