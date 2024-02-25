# Generates data for inside and outside the unit disc

import numpy as np
import pandas as pd

x_1 = np.random.uniform(-2, 2, 200)
x_2 = np.random.uniform(-2, 2, 200)
y = 1 * (np.sqrt(x_1**2 + x_2**2) > 1)

df = pd.DataFrame({'x_1': x_1, 'x_2': x_2, 'y': y})
df.to_csv("circle_data.csv")
