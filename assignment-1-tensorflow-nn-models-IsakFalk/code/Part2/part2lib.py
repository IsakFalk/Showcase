### library for code in part 2 of assignment 1

import sys
import os
import shutil
import random
import pickle

import tensorflow as tf
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.examples.tutorials.mnist import input_data

# seed the thing
np.random.seed(100)

### helper functions

def weight(num_rows, num_cols):
    """Get a weight matrix for the specified input and output size

    Use Xavier initalisation"""

    W = np.random.normal(loc=0.0, scale=1.0/np.sqrt(num_cols), size=(num_rows, num_cols)).astype(np.float32)

    return W


def bias(size):
    """Get a bias vector for the specified output size

    Initialise it to be slightly positive constant to avoid
    dead units"""

    b = 0.1*np.ones((size, 1), dtype=np.float32).reshape(size, 1)

    return b

### Classes needed

# All numbers will be 32 bit floating point number (np.float32)
# except for in the case of int

class Data():
    """Implements a data class for holding training and test set

    Since we work with column vectors, need to be careful so that we
    convert from row form that tensorflow gives us"""

    def __init__(self, batch_size):
        """Get the mnist data using tf datasets"""

        # get the mnist data set
        mnist = input_data.read_data_sets("./../../data", one_hot=True)

        # Transpose as we are working with column vectors
        self.train_images = mnist.train.images.T
        self.train_labels = mnist.train.labels.T
        self.train_size = np.shape(mnist.train.labels)[0]

        self.test_images = mnist.test.images.T
        self.test_labels = mnist.test.labels.T
        self.test_size = np.shape(mnist.test.labels)[0]

        self.val_images = mnist.validation.images.T
        self.val_labels = mnist.validation.labels.T
        self.val_size = np.shape(mnist.validation.labels)[0]

        # Assert that the batch size is a divisor of the number of training samples
        assert self.train_size % batch_size == 0

        self.batch_size = batch_size
        self.batch_count = 0
        self.epoch = 0

        self.random_index = np.random.permutation(self.train_size)

    def next_batch(self):
        """Get the next batch for doing SGD"""

        i = self.batch_count
        s = self.batch_size

        batch_images = self.train_images[:, self.random_index[i*s:(i + 1)*s]]
        batch_labels = self.train_labels[:, self.random_index[i*s:(i + 1)*s]]

        # increment batch_count, if at end, make 0
        self.batch_count += 1

        # full epoch, reset attributes needed for next epoch
        if self.batch_count*self.batch_size == self.train_size:
            self.epoch += 1
            self.batch_count = 0
            self.random_index = np.random.permutation(self.train_size)

        return batch_images, batch_labels

    def set_batch_size(self, size):
        """Reset the batch size for reusing the data"""

        # Assert that the batch size is a divisor of the number of training samples
        assert self.train_size % size == 0

        self.batch_size = size

    def restart(self):
        """Restart the counter so that we can reuse
        the same data instance for several models in a row"""

        self.epoch = 0
        self.batch_count = 0
        self.random_index = np.random.permutation(self.train_size)


class Layer():
    """Base class for all kinds of layer"""

    def __init__(self, ):
        "docstring"
        pass

    # Abstract
    def forward_pass(self):
        print 'implement the forward pass'

    def backward_pass(self):
        print 'implement the backward pass'


class ReLU(Layer):
    """ReLU layer"""

    def __init__(self):
        pass

    def set_variables(self, x, y):
        """Set the input and output to the layer for the current iteration"""

        self.x = x
        self.y = y

    @staticmethod
    def forward_pass(x):
        """Get output from layer"""

        y = np.maximum(x, 0, x)

        assert y.shape == x.shape

        return y

    def backward_pass(self, learning_rate, dL_dy, y_):
        """Get backwards pass dL_dx"""

        dL_dy[self.x.T < 0] = 0
        dL_dx = dL_dy

        # reset variables to zero
        self.x = 0
        self.y = 0

        return dL_dx


class Softmax(Layer):
    """Softmax layer module"""

    def __init__(self):
        pass

    def set_variables(self, x, y):
        """Set the input and output to the layer for the current iteration"""

        self.x = x
        self.y = y

    @staticmethod
    def forward_pass(x):
        """Calculate the elementwise sigmoid of vector x"""

        e_x = np.exp(x - np.max(x))

        assert e_x.shape == x.shape

        return e_x / e_x.sum(axis=0)

    def backward_pass(self, learning_rate, dL_dy, y_):
        """Calculate the backward pass"""

        # reset variables to zero
        self.x = 0
        self.y = 0

        pass


class LinearLayer(Layer):
    """linear layer module

    W: output_size x input_size
    b: output_size x 1"""

    def __init__(self, input_size, output_size):
        """Initialise all of the needed variables for the layers"""

        self.input_size = input_size
        self.output_size = output_size

        self.W = weight(output_size, input_size)
        self.b = bias(output_size)

    def set_variables(self, x, y):
        """Set the input and output to the layer for the current iteration"""

        self.x = x
        self.y = y

    def set_parameters(self, W, b):
        """Set the parameters W and b

        Mainly used to load saved models for
        evaluating their score. A kind of loading
        mechanism."""

        assert self.W.shape == W.shape, "Passed W not correct shape"
        assert self.b.shape == b.shape, "Passed b not correct shape"

        self.W = W
        self.b = b

    def forward_pass(self, x):
        """Get output from layer"""

        return np.dot(self.W, x) + self.b

    def backward_pass(self, learning_rate, dL_dy, y_):
        """Backward pass"""
        dL_dx = np.dot(dL_dy, self.W)

        self.b = self.b - learning_rate*np.sum(dL_dy.T, 1).reshape(self.b.shape)
        self.W = self.W - learning_rate*(np.dot(self.x, dL_dy)).T

        # reset variables to zero
        self.x = 0
        self.y = 0

        return dL_dx

    def param_gradients(self, dL_dy):
        """Gradients of the parameters"""

        self.dL_db = dL_dy
        self.dL_dW = np.dot(self.x, dL_dy)


class OutputLayer(Layer):
    """Softmax layer"""

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.W = weight(output_size, input_size)
        self.b = bias(output_size)

    def set_parameters(self, W, b):
        """Set the parameters W and b

        Mainly used to load saved models for
        evaluating their score. A kind of loading
        mechanism."""

        assert self.W.shape == W.shape, "Passed W not correct shape"
        assert self.b.shape == b.shape, "Passed b not correct shape"

        self.W = W
        self.b = b

    def set_variables(self, x, y):
        """Set then input and output to the layer for the current iteration"""

        self.x = x
        self.y = y

    def softmax(self, z):
        """Calculate the elementwise sigmoid of vector x"""

        # we make it more stable by taking away the maximum
        # (might want to add an offset)

        y = np.exp(z - np.max(z))
        y = y / y.sum(axis=0)

        return y

    def logits(self, x):
        """Return the logits to be used when calculating the cross entropy directly"""

        z = np.dot(self.W, x) + self.b

        return z

    def Xent_from_logits(self, z, y_):
        """Calculate the cross entropy directly.

        We follow the tensorflow implementation
        in order to avoid numerical issues

        See: tf.nn.sigmoid_cross_entropy_with_logits.md on github for reference"""
        cross_entropy = np.max(z) - np.dot(y_.T, z) + np.log(np.sum(np.exp(z - np.max(z))))

        return cross_entropy

    def forward_pass(self, x):
        """Get output from layer"""

        y = self.softmax(self.logits(x))

        return y

    def backward_pass(self, learning_rate, dL_dy, y_):
        """Backward pass"""

        dL_dx = np.dot((self.y - y_).T, self.W)

        self.b = self.b - learning_rate*np.sum(self.y - y_, 1).reshape(self.b.shape)
        self.W = self.W - learning_rate*np.dot((self.y - y_), self.x.T)

        return dL_dx

    def param_gradients(self, dL_dy, y_):
        """Gradients for the parameters"""

        self.dL_db = (self.y - y_).T
        self.dL_dW = np.dot(self.x, (self.y - y_).T)


class NeuralNetwork(object):
    """A high level-class for handling the layers

    NeuralNetwork will handle training, prediction and passing
    of gradients. It will handle the layers by keeping track of
    a list: [layer1, layer2, layer3] etc. It will only do on-line
    SGD."""

    def __init__(self, layers, learning_rate, train_epochs, batch_size, name):
        self.layers = layers
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.name = name
        self.batch_size = batch_size
        self.dL_dy = 1

    def feed_forward(self, x, accuracy=False):
        """Get the output from the neural network given the input x"""

        # placeholder values that we will pass on and save in each layer
        x_ = x
        y = x_

        # we start from the input and go all the way until the final layer
        for i in range(len(self.layers)):
            y = self.layers[i].forward_pass(x_)
            if not accuracy:
                self.layers[i].set_variables(x_, y)
            x_ = y

        return y

    def feed_backward(self, y, y_):
        """Feed backward and update parameters in the same round"""
        dL_dy = 1

        for i in reversed(range(len(self.layers))):
            dL_dy = self.layers[i].backward_pass(self.learning_rate, dL_dy, y_)
            self.layers[i].set_variables(None, None)

        return dL_dy

    def get_cost(self, x, y_):
        """Slightly different if we also need to get the cost"""

        # we start from the input and go all the way until the final layer
        # there we apply the direct to Xent to avoid some calculations
        y = x

        # feed forward until we are at the input to the last layer
        for i in range(len(self.layers) - 1):
            y = self.layers[i].forward_pass(y)

        # make the logits and calculate the Xentropy from this
        # output and the true output
        y = self.layers[-1].logits(y)
        cost = self.layers[-1].Xent_from_logits(y, y_)

        return cost

    def accuracy(self, type_set, data):
        """Get the accuracy of the set and the output prediction matrix

        The prediction matrix is used to get the needed predicted
        digits to be able to calculate the confusion matrix"""

        if type_set.lower() == "test":
            images = data.test_images
            labels = data.test_labels
            set_size = data.test_size
        elif type_set.lower() == "train":
            images = data.train_images
            labels = data.train_labels
            set_size = data.train_size
        elif type_set.lower() == "val":
            images = data.val_images
            labels = data.val_labels
            set_size = data.val_size

        a1 = np.argmax(self.feed_forward(images, accuracy=True), axis=0)
        a2 = np.argmax(labels, axis=0)

        mean_accuracy = np.mean((a1 == a2).astype(np.float32))

        return mean_accuracy

    def train(self, data, test):
        """Train the model

        We record several lists that will hold accuracy
        of the model as we train. it, test_acc, train_acc is
        mainly used for plotting, while epoch_acc is for
        hyperparameter optimisation. set test to 'val' for
        doing hyperparameter optimisation.

        keywords:
        data: instance of the Data class holding the data
        test: specify if we should use the test or validation set"""

        # Need to reset and train on the data
        data.restart()
        data.set_batch_size(self.batch_size)

        # arrays for plotting

        it = []
        test_acc = []
        train_acc = []
        epoch_acc = []

        while data.epoch < self.train_epochs:
            for i in range(data.train_size/self.batch_size):

                # Get the next data point
                x, y_ = data.next_batch()

                # Feed it forward
                y = self.feed_forward(x)

                # Feed it backward and update parameters
                dL_dy = self.feed_backward(y, y_)

                # record accuracy at every 10000th step
                if i % 100 == 0:
                    it.append(data.epoch*data.train_size/self.batch_size + i)
                    train_acc.append(self.accuracy('train', data))
                    test_acc.append(self.accuracy(test, data))

            print "Epoch: {}, Train error: {:.5f}, {} error: {:.5f}".format(data.epoch, 1 - self.accuracy('train', data), test,  1 - self.accuracy(test, data))
            epoch_acc.append(self.accuracy(test, data))

        return it, test_acc, train_acc, epoch_acc

    def plot_convergence(self, it, test_acc, train_acc, name):
        """Plot and save the convergence graphs for the accuracy"""

        os.chdir('./../../report/img/')

        try:
            os.remove('./' + name + '.eps')
        except OSError:
            pass

        sns.set_style('darkgrid')

        fig = plt.figure()

        train_err = list(1 - np.array(train_acc))
        test_err = list(1 - np.array(test_acc))

        ymax_train = np.max(train_err)
        ymax_test = np.max(test_err)
        ymax = np.max([ymax_train, ymax_test])

        plt.plot(it, train_err, 'b', label='Train', linewidth=1.5)
        plt.plot(it, test_err, 'r', label='Test', linewidth=1.5)
        plt.title('Accuracy over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.ylim([0, ymax])
        plt.legend()

        plt.savefig('error_' + name + '.eps', format='eps', dpi=1000)
        plt.clf()

        print 'Plot ' + name + '.eps' + ' saved in image folder'

        os.chdir('./../../code/Part2')

    def plot_confusion_matrix(self, data, name):
        """Plot and save a confusion matrix of the MNIST data set using NN

        keywords:
        name: name which we will save the plot as
        predictions: the predicted digit in list form
        true_output: the true digit in list form

        Taken from: http://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib"""

        os.chdir('./../../report/img/')

        # We need to create a matrix filled out with the empirical probabilities of
        # each entry. The entry (i,j) is the normalised counts of predicting digit i
        # given that the true digit is j.

        images = data.test_images
        labels = data.test_labels

        predictions = np.argmax(self.feed_forward(images, True), axis=0)
        true_output = np.argmax(labels, axis=0)

        assert len(predictions) == len(true_output)

        conf_mat = np.zeros((10, 10)).astype(np.int32)

        for i in range(len(predictions)):
            conf_mat[true_output[i]][predictions[i]] += 1

        fig, ax = plt.subplots()
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(np.ones((10, 10)))

        ticks = np.arange(0, 10, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        for (i, j), z in np.ndenumerate(conf_mat):
            ax.text(j, i, '{:d}'.format(z), ha='center', va='center', size=9)

        plt.savefig(name + '.eps', format='eps', dpi=100, facecolor='white')
        plt.clf()

        print 'Confusion matrix ' + name + '.eps' + ' saved in image folder'

        os.chdir('./../../code/Part2')


def save_network(network):
    """Save the learned parameters of the model

    keywords:
    network: a NeuralNetwork instance

    rel_path specifies the relative path where to save
    the parameters, (W, b), for each linear layer.

    We use the python pickle format to serialize the whole
    Neural Network data structure."""

    rel_path = './../../save/np/' + network.name + ".pickle"

    with open(rel_path, 'wb') as f:
        pickle.dump(network, f)

    print network.name + ' saved to ' + rel_path


def load_network(file_name):
    """Load the pickled network with the learned parameters

    the file_name has to correspond to the file name of the
    pickled network and have to be in the save/np/ directory
    relative to the top directory"""

    rel_path = './../../save/np/' + file_name + ".pickle"

    with open(rel_path, 'rb') as f:
        network = pickle.load(f)

    return network


def optimize_hyperparams():
    """Optimize the hyperparameter using random search

    Due to the sensitivity of the Neural Nets with regard
    to the learning rate the set of hyperparameters will be
    be tailored for each model. We will take the number of
    samples to be 4 due to the long time of the training of
    the models. We will perform grid search over the learning
    rate, and it will be each model to avoid blow-up. We will do 150
    epochs and record the argmax of the accuracy of each run
    in order to yield the optimal stopping time of each model. the
    batch size will be 200"""

    data = Data(1)

    # model 2a

    print "model 2a"

    input_size = 784
    output_size = 10

    # hyperparameter set
    ep = 150
    learning_rate = [0.1*10**(-i) for i in range(3)]
    batch_size = 200

    # optimal choices so far
    opt = [0, 0, 0]
    opt_acc = 0

    for lr in learning_rate:
        print "Learning rate: {}".format(lr)

        layers = [OutputLayer(input_size, output_size)]
        model_2a = NeuralNetwork(layers, lr, ep, batch_size, 'model_2a')
        _, _, _, epoch_acc = model_2a.train(data, 'val')
        current_acc = np.max(epoch_acc)
        epoch_ind = np.argmax(epoch_acc) + 1

        current = [lr, epoch_ind, current_acc] 

        print "[Learning rate, optimal epoch, optimal error] for this run: [{}, {}, {}]".format(lr, epoch_ind, 1 - current_acc)

        if current_acc > opt_acc:
            opt = current
            opt_acc = current_acc

    print "Optimal hyperparameters: [{}, {}], Validation error: {}".format(opt[0], opt[1], 1 - opt[2])

    # model 2b

    print "model 2b"

    input_size = 784
    output_size = 10
    hidden_size = 128

    # hyperparameter set
    ep = 300
    learning_rate = [0.001*10**(-i) for i in range(3)]
    batch_size = 200

    # optimal choices so far
    opt = [0, 0, 0]
    opt_acc = 0

    for lr in learning_rate:
        print "Learning rate: {}".format(lr)

        layers = [LinearLayer(input_size, hidden_size),
                  ReLU(),
                  OutputLayer(hidden_size, output_size)]
        model_2b = NeuralNetwork(layers, lr, ep, batch_size, 'model_2b')
        _, _, _, epoch_acc = model_2b.train(data, 'val')
        current_acc = np.max(epoch_acc)
        epoch_ind = np.argmax(epoch_acc) + 1

        current = [lr, epoch_ind, current_acc]

        print "[Learning rate, optimal epoch, optimal error] for this run: [{}, {}, {}]".format(lr, epoch_ind, 1 - current_acc)

        if current_acc > opt_acc:
            opt = current
            opt_acc = current_acc

    print "Optimal hyperparameters: [{}, {}], Validation error: {}".format(opt[0], opt[1], 1 -  opt[2])

    # model 2c

    print "model 2c"

    input_size = 784
    output_size = 10
    hidden_size = 256

    # hyperparameter set
    ep = 500
    learning_rate = [0.0001*10**(-i) for i in range(3)]
    batch_size = 200

    # optimal choices so far
    opt = [0, 0, 0]
    opt_acc = 0

    for lr in learning_rate:
        print "Learning rate: {}".format(lr)

        layers = [LinearLayer(input_size, hidden_size),
                  ReLU(),
                  LinearLayer(hidden_size, hidden_size),
                  ReLU(),
                  OutputLayer(hidden_size, output_size)]

        model_2c = NeuralNetwork(layers, lr, ep, batch_size, 'model_2c')
        _, _, _, epoch_acc = model_2c.train(data, 'val')
        current_acc = np.max(epoch_acc)
        epoch_ind = np.argmax(epoch_acc) + 1

        current = [lr, epoch_ind, current_acc]

        print "[Learning rate, optimal epoch, optimal error] for this run: [{}, {}, {}]".format(lr, epoch_ind, 1 - current_acc)

        if current_acc > opt_acc:
            opt = current
            opt_acc = current_acc

    print "Optimal hyperparameters: [{}, {}], Validation error: {}".format(opt[0], opt[1], 1 - opt[2])

    # model 2d


if __name__ == '__main__':
    optimize_hyperparams()
