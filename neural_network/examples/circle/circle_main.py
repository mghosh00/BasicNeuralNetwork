import pandas as pd

from neural_network import Network
from neural_network import Trainer

circle_data = pd.read_csv("circle_data.csv")
network = Network(2, 2, [4, 4],
                  learning_rate=0.03)
trainer = Trainer(network, circle_data, 1000, 10)
trainer.train()
trainer.generate_plot()
