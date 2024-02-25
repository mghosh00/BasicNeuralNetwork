# Generates data for inside and outside the unit disc

import numpy as np
import pandas as pd

x_1 = np.random.uniform(-1, 1, 200)
x_2 = np.random.uniform(-1, 1, 200)
y = 1 * (x_1 > x_2)

df = pd.DataFrame({'x_1': x_1, 'x_2': x_2, 'y': y})
df.to_csv("linear_data.csv")
