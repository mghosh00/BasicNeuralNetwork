from neural_network import DataSplitter
from neural_network import Network
from neural_network import Validator
from neural_network import Trainer
from neural_network import Tester

# Splitting our data
data_splitter = DataSplitter("mean_data.csv", [8, 1, 1])
training_data, validation_data, testing_data = data_splitter.split()

# Setting up the neural network
learning_rate = 0.001
network = Network(2, 2, [4, 4],
                  regression=True, learning_rate=learning_rate,
                  adaptive=True, he_weights=True)

# Creating the different phases of learning
validator = Validator(network, validation_data, 10)
num_epochs = 100
trainer = Trainer(network, training_data, num_epochs, 16,
                  weighted=True, validator=validator)
tester = Tester(network, testing_data, 10)

# Running the trainer and validator and generating some plots
trainer.run()
trainer.generate_loss_plot(title=f'mean_epochs_{num_epochs}')
trainer.generate_scatter(title=f'mean_epochs_{num_epochs}')
trainer.comparison_scatter(title=f'mean_epochs_{num_epochs}')
validator.generate_scatter(title=f'mean_epochs_{num_epochs}')

# Running the tester and generating some plots
tester.run()
tester.generate_scatter(title=f'mean_epochs_{num_epochs}')
