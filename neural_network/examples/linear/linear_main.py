import pandas as pd

from neural_network import Network
from neural_network import Trainer

linear_data = pd.read_csv("linear_data.csv")
network = Network(2, 2, [4, 4])
trainer = Trainer(network, linear_data, 1000, 10)
trainer.train()
trainer.generate_plot()
