# Import all of the needed libraries

import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.examples.tutorials.mnist import input_data

# Import dataset with one-hot class encoding, the directory is simply data
mnist = input_data.read_data_sets("data", one_hot=True)

# Seed the RNG of Tensorflow and Numpy
seed = sum(map(ord, 'Assignment1'))
tf.set_random_seed(seed)
np.random.seed(seed)

# Functions
def plot_convergence(name, train_acc, test_acc, it):
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

    plt.savefig(name + '.eps', format='eps', dpi=1000)
    plt.clf()

    print 'Plot ' + name + '.eps' + ' saved in image folder'


    os.chdir('./../../code/Part1')


def plot_confusion_matrix(name, predictions, true_output):
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

    assert len(predictions) == len(true_output)

    conf_mat = np.zeros((10, 10)).astype(np.int32)

    for i in range(len(predictions)):
        conf_mat[true_output[i]][predictions[i]] += 1

    # # normalise it for every digit row
    # for i in range(10):
    #     conf_mat[i, :] = conf_mat[i, :] / np.sum(conf_mat[i, :])

    fig, ax = plt.subplots()
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(np.ones((10, 10)))

    ticks = np.arange(0, 10, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    for (i, j), z in np.ndenumerate(conf_mat):
        ax.text(j, i, '{}'.format(z), ha='center', va='center', size=9)

    plt.savefig(name + '.eps', format='eps', dpi=100, facecolor='white')
    plt.clf()

    os.chdir('./../../code/Part1')


def get_placeholders():
    """Define the placeholders to be used for x and y"""

    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    return x, y_


def train_model(sess, x, y_, train_set, test_set, train_step, batch_size, epochs, accuracy, test='Test'):
    """Train the model through SGD"""

    # iteration
    it = []

    # lists for holding accuracy values
    train_acc = []
    test_acc = []
    epoch_acc = []

    # Need to split up the train set into two sets so that
    # my computer can handle calculating the accuracy.
    n_train = train_set.labels.shape[0]
    n_traind2 = int(np.floor(n_train/2.0))

    train_images1 = train_set.images[:n_traind2, :]
    train_labels1 = train_set.labels[:n_traind2, :]

    train_images2 = train_set.images[n_traind2:, :]
    train_labels2 = train_set.labels[n_traind2:, :]

    # train the model to learn optimal parameters
    num_epochs = 0

    while num_epochs < epochs:
        for i in range(n_train/batch_size):
            batch_xs, batch_ys = train_set.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % 100 == 0:
                it.append(num_epochs*n_train/batch_size + i)
                train_acc_temp = (sess.run(accuracy,
                                           feed_dict={x: train_images1,
                                                      y_: train_labels1})
                                  + sess.run(accuracy,
                                             feed_dict={x: train_images2,
                                                        y_: train_labels2}))/2.0
                train_acc.append(train_acc_temp)
                test_acc.append(sess.run(accuracy, feed_dict={x: test_set.images, y_: test_set.labels}))

        num_epochs += 1

        train_acc_temp = (sess.run(accuracy,
                                   feed_dict={x: train_images1,
                                              y_: train_labels1})
                          + sess.run(accuracy,
                                     feed_dict={x: train_images2,
                                                y_: train_labels2}))/2.0

        train_err = 1 - train_acc_temp
        test_err = 1 - sess.run(accuracy, feed_dict={x: test_set.images,
                                                       y_: test_set.labels})

        print "Epoch: {}, Train error: {:.5f}, {} error: {:.5f}".format(num_epochs, train_err, test, test_err)
        epoch_acc.append(1 - test_err)

    return train_acc, test_acc, epoch_acc, it


def save_to_disk(sess, model_name, accuracy, x, y_, y):
    """Save the current model and variables to disk"""

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y_', y_)
    tf.add_to_collection('y', y)

    abs_path = os.path.join(os.getcwd(), '..', '..', 'save', 'tf', model_name, 'model_1' + model_name + ".ckpt")
    save_path = saver.save(sess, abs_path)

    print("Model saved in file: {}".format(save_path))


def start_session():
    """Start the session and initialize all of the variables"""

    # Initialise the session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    return sess


def accuracy_op(y, y_):
    """Get the accuracy operation"""

    # Get a logical vector where a 1 mean that the classification was correct, 0 wrong

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # Get the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def weight_variable(shape, name):
    """Create a weight variable"""

    # Xavier initalisation
    initial = tf.truncated_normal(shape, stddev=(1.0/np.sqrt(748)))
    return tf.Variable(initial)


def bias_variable(shape, name):
    """Create a bias variable"""

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """Convolve the images"""

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """Create a max pool for the image vectors"""

    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def optimize_hyperparams(run_model_func):
    """Use random grid search over learning rate. We validate that the test
    error doesn't increase. The batch size is set to 200 for all models. We run
    it for 50 epochs.

    We do grid search on the set [0.1, 0.01, 0.001, 0.0001]

    Validation set used to validate the hyperparameter.
    """

    validation_set = mnist.validation
    epochs = 100
    batch_size = 200

    # optimal choices so far
    opt = [0, 0, 0]
    opt_acc = 0

    l_r = [10**(-i) for i in range(1, 5)]

    for learn_rate in l_r:
        print "Current learning rate {}\n".format(learn_rate)

        _, train_acc, test_acc, it, epoch_acc,  _ = run_model_func(batch_size, epochs, learn_rate, validation_set)

        current_acc = np.max(epoch_acc)
        epoch_ind = np.argmax(epoch_acc) + 1

        current = [learn_rate, epoch_ind, current_acc]

        print "[Learning rate, optimal epoch, optimal error] for this run: [{}, {}, {}]\n".format(learn_rate, epoch_ind, 1 - current_acc)

        if current_acc > opt_acc:
            opt = current
            opt_acc = current_acc

    print "Optimal hyperparameters: [{}, {}], Validation error: {}\n".format(opt[0], opt[1], 1 - opt[2])

    return opt
