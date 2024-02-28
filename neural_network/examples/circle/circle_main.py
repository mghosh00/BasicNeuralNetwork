from neural_network import DataSplitter
from neural_network import Network
from neural_network import Validator
from neural_network import Trainer
from neural_network import Tester

# Splitting our data
data_splitter = DataSplitter("circle_data.csv", [8, 1, 1])
training_data, validation_data, testing_data = data_splitter.split()

# Setting up the neural network
learning_rate = 0.01
network = Network(2, 2, [4, 4],
                  learning_rate=learning_rate, adaptive=True,
                  he_weights=True)

# Creating the different phases of learning
validator = Validator(network, validation_data, 10)
trainer = Trainer(network, training_data, 1000, 16,
                  weighted=True, validator=validator)
tester = Tester(network, testing_data, 10)

# Running the trainer and validator and generating some plots
trainer.run()
trainer.generate_loss_plot(title=f'circle_lr_{learning_rate}_he')
trainer.generate_scatter(title=f'circle_lr_{learning_rate}_he')
validator.generate_scatter(title=f'circle_lr_{learning_rate}_he')

# Running the tester and generating some plots
tester.run()
tester.generate_confusion()
tester.generate_scatter(title=f'circle_lr_{learning_rate}_he')
