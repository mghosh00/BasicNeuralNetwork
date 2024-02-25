import pandas as pd

from neural_network import Network
from neural_network import Validator
from neural_network import Trainer
from neural_network import Tester

data = pd.read_csv("circle_data.csv")
training_data = data.head(200)
validation_data = data.tail(100).head(50)
testing_data = data.tail(50)
validation_data.index = list(range(50))
testing_data.index = list(range(50))

learning_rate = 0.01
network = Network(2, 2, [4, 4],
                  learning_rate=learning_rate, adaptive=True)

validator = Validator(network, validation_data, 10)
trainer = Trainer(network, training_data, 1000, 16,
                  weighted=True, validator=validator)
tester = Tester(network, testing_data, 10)

trainer.run()
trainer.generate_loss_plot(title=f'circle_lr_{learning_rate}')
validator.generate_scatter(title=f'circle_lr_{learning_rate}')

tester.run()
tester.generate_confusion()
