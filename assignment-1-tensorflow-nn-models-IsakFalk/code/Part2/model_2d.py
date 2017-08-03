# Model 2d (Not working, removed code in part2lib that was faulty)

import part2lib as p2l

# define the layers needed for the models

input_size = 784
output_size = 10
batch_size = 200
img_width = 28
img_height = 28
channels = 1
filters = 16

# shape of the batched images, resized
image_shape = (batch_size, channels, img_height, img_width)
filter_shape = (batch_size, filters, img_height/2, img_width/2)
flatten_output = (batch_size, filters * img_height/4 * img_width/4)

layers = [p2l.reshape_input(),
          p2l.conv2d(image_shape),
          p2l.max_pooling(),
          p2l.conv2d(filter_shape),
          p2l.max_pooling(),
          p2l.flatten(),
          p2l.ReLU(),
          p2l.OutputLayer(16 * 28/4 * 28/4, 10)]

# build the network
# Hyperparameters

learning_rate = 1e-4
train_epochs = 175

data = p2l.Data(batch_size)

# Define the model
model_2d = p2l.NeuralNetwork(layers, learning_rate, train_epochs, batch_size, 'model_2d')

# Train it and get the lists for plotting
it, test_acc, train_acc, epoch_acc = model_2d.train(data, 'test')

# Plot the errors and get the confusion matrix
model_2d.plot_convergence(it, test_acc, train_acc, 'model_2d')
model_2d.plot_confusion_matrix(data, 'model_2d_confusion_matrix')

# Save the model to disk
p2l.save_network(model_2d)
