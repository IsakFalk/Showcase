# Model 2a

import part2lib as p2l

# define the layers needed for the models

input_size = 784
output_size = 10

layers = [p2l.OutputLayer(input_size, output_size)]

# build the network

# optimal hyperparameters

learning_rate = 1e-3
train_epochs = 107
batch_size = 200

data = p2l.Data(1)

# Define the model
model_2a = p2l.NeuralNetwork(layers, learning_rate, train_epochs, batch_size, 'model_2a')

# Train it and get the lists for plotting
it, test_acc, train_acc, epoch_acc = model_2a.train(data, 'test')

# Plot the errors and get the confusion matrix
model_2a.plot_convergence(it, test_acc, train_acc, 'model_2a')
model_2a.plot_confusion_matrix(data, 'model_2a_confusion_matrix')

# Save the model to disk
p2l.save_network(model_2a)
