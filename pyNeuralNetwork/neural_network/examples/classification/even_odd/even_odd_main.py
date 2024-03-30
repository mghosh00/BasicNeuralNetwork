from neural_network import DataSplitter
from neural_network import Network
from neural_network import Validator
from neural_network import Trainer
from neural_network import Tester

# Splitting our data
data_splitter = DataSplitter("even_odd_data.csv", [8, 1, 1])
training_data, validation_data, testing_data = data_splitter.split()

# Setting up the neural network
learning_rate = 0.01
network = Network(2, 3, [7, 7, 7],
                  learning_rate=learning_rate, num_classes=2, adaptive=True,
                  he_weights=True)

# Creating the different phases of learning
validator = Validator(network, validation_data, 10)
num_epochs = 4000
trainer = Trainer(network, training_data, num_epochs, 16,
                  weighted=True, validator=validator)
tester = Tester(network, testing_data, 10)

# Running the trainer and validator and generating some plots
trainer.run()
trainer.generate_loss_plot(title=f'even_odd_epochs_{num_epochs}')
trainer.generate_scatter(title=f'even_odd_epochs_{num_epochs}')
validator.generate_scatter(title=f'even_odd_epochs_{num_epochs}')

# Running the tester and generating some plots
tester.run()
tester.generate_confusion()
tester.generate_scatter(title=f'even_odd_epochs_{num_epochs}')
