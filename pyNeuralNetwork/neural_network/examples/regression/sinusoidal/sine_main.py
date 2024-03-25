from neural_network import DataSplitter
from neural_network import Network
from neural_network import Validator
from neural_network import Trainer
from neural_network import Tester

# Splitting our data
data_splitter = DataSplitter("sine_data.csv", [8, 1, 1])
training_data, validation_data, testing_data = data_splitter.split()

# Setting up the neural network
learning_rate = 0.001
network = Network(2, 3, [6, 6, 6],
                  regression=True, learning_rate=learning_rate,
                  adaptive=True, he_weights=True)

# Creating the different phases of learning
validator = Validator(network, validation_data, 10)
num_epochs = 1000
trainer = Trainer(network, training_data, num_epochs, 16,
                  weighted=True, validator=validator)
tester = Tester(network, testing_data, 10)

# Running the trainer and validator and generating some plots
title = f'sine_epochs_{num_epochs}'
trainer.run()
trainer.generate_loss_plot(title=title)
trainer.generate_scatter(title=title)
trainer.comparison_scatter(title=title)
validator.generate_scatter(title=title)

# Running the tester and generating some plots
tester.run()
tester.generate_scatter(title=title)
tester.comparison_scatter(title=title)
