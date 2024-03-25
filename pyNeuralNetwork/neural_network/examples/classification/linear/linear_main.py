import pandas as pd

from neural_network import Network
from neural_network import Validator
from neural_network import Trainer
from neural_network import Tester

data = pd.read_csv("linear_data.csv").iloc[:, 1:]
training_data = data.head(200)
validation_data = data.tail(100).head(50)
testing_data = data.tail(50)
validation_data.index = list(range(50))
testing_data.index = list(range(50))

network = Network(2, 2, [4, 4])
validator = Validator(network, validation_data, 5)
trainer = Trainer(network, training_data, 1000, 10,
                  validator=validator)
tester = Tester(network, testing_data, 5)

trainer.run()
trainer.generate_loss_plot()
trainer.generate_scatter()
validator.generate_scatter()

tester.run()
tester.generate_confusion()
tester.generate_scatter()
