# Model 2c

import part2lib as p2l

# define the layers needed for the models

input_size = 784
output_size = 10
hidden_size = 256

layers = [p2l.LinearLayer(input_size, hidden_size),
          p2l.ReLU(),
          p2l.LinearLayer(hidden_size, hidden_size),
          p2l.ReLU(),
          p2l.OutputLayer(hidden_size, output_size)]

# build the network
# Hyperparameters

learning_rate = 1e-4
train_epochs = 175
batch_size = 200

data = p2l.Data(batch_size)

# Define the model
model_2c = p2l.NeuralNetwork(layers, learning_rate, train_epochs, batch_size, 'model_2c')

# Train it and get the lists for plotting
it, test_acc, train_acc, epoch_acc = model_2c.train(data, 'test')

# Plot the errors and get the confusion matrix
model_2c.plot_convergence(it, test_acc, train_acc, 'model_2c')
model_2c.plot_confusion_matrix(data, 'model_2c_confusion_matrix')

# Save the model to disk
p2l.save_network(model_2c)
