"""
Utility functions that has to do with tensorflow for task 1.
"""

import os

import numpy as np
import tensorflow as tf

import preprocess
from SETTINGS import *


class Data():
    """ Class for holding the data and binarizing it."""

    def __init__(self, batch_size):
        self.data = preprocess.get_mnist()
        self.batch_size = batch_size
        self.train_size = self.data.train.labels.shape[0]
        self.test_size = self.data.test.labels.shape[0]

    def get_train_batch(self):
        """ Get a new training batches"""
        x_batch, y_batch = self.data.train.next_batch(self.batch_size)
        x_batch = preprocess.binarize(x_batch)
        return x_batch, y_batch

    def get_test_batch(self):
        """ Get a new test batches"""
        x_batch, y_batch = self.data.test.next_batch(self.batch_size)
        x_batch = preprocess.binarize(x_batch)
        return x_batch, y_batch

    def get_random_train_batch(self, size):
        """ Get random train batch"""
        random_ind_train = np.random.permutation(self.train_size)[0:size]
        x_rand_batch = preprocess.binarize(self.data.train.images[random_ind_train, :])
        y_rand_batch = self.data.train.labels[random_ind_train, :]
        return x_rand_batch, y_rand_batch

    def get_random_test_batch(self, size):
        """ Get random test batch"""
        random_ind_test = np.random.permutation(self.test_size)[0:size]
        x_rand_batch = preprocess.binarize(self.data.test.images[random_ind_test, :])
        y_rand_batch = self.data.test.labels[random_ind_test, :]
        return x_rand_batch, y_rand_batch

    def get_normal_test_batch(self):
        return self.data.test.next_batch(self.batch_size)

    def get_normal_train_batch(self):
        return self.data.train.next_batch(self.batch_size)

    def get_normal_random_test_batch(self, size):
        random_ind_train = np.random.permutation(self.test_size)[0:size]
        x_rand_batch = self.data.test.images[random_ind_train, :]
        y_rand_batch = self.data.test.labels[random_ind_train, :]
        return x_rand_batch, y_rand_batch

    def get_normal_random_train_batch(self, size):
        random_ind_train = np.random.permutation(self.train_size)[0:size]
        x_rand_batch = self.data.train.images[random_ind_train, :]
        y_rand_batch = self.data.train.labels[random_ind_train, :]
        return x_rand_batch, y_rand_batch

    def get_train_data(self):
        """ Get the whole training data set, images and labels"""
        images = preprocess.binarize(self.data.train.images)
        labels = self.data.train.labels
        return images, labels

    def get_test_data(self):
        """ Get the whole test data set, images and labels"""
        images = preprocess.binarize(self.data.test.images)
        labels = self.data.test.labels
        return images, labels


def debug(sess, *args):
    """ Debug by evaluating the tensor, getting the
    numpy array.

    Args:
    sess: (tf.Session) current session
    args: (*tf.tensor) tensors to evaluate"""

    args = list(args)
    for arg in args:
        print "tensor: {}\nshape: {}\n".format(str(arg), str(arg.get_shape()))


def reset_graph(sess):
    """ Resets the graph in use. """

    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def batch_norm_wrapper(inputs, phase, decay=0.99, epsilon=0.001):
    """ Performs the internals needed to do batch norm.

    Args:
    inputs: (tf.tensor) tensor that we perform batch normalization on
    phase: (tf.bool) boolean signifying if we are in train or test phase
    decay: (float) the decay that we run the exponential average of the
    test mean and var. Higher decay implies more iterations until convergence
    of pop_mean and pop_var but that they in general will be more accurate.
    epsilon: (float) regularisation parameter to avoid 0division

    Returns:
    batch_norm_outputs: (tf.tenser) tensor of the batch normalized inputs

    NOTE: This function is taken from the guide
    http://r2rt.com/implementing-batch-normalization-in-tensorflow.html"""

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    # Functions to pass to tf conditional
    def is_training():
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean,
                                             batch_var,
                                             beta,
                                             scale,
                                             epsilon)

    def is_test():
        return tf.nn.batch_normalization(inputs,
                                         pop_mean,
                                         pop_var,
                                         beta,
                                         scale,
                                         epsilon)

    return tf.cond(phase, is_training, is_test)


def summary_ops(out_dir, name, timestamp, loss, accuracy):
    """ Summary operations (task 1).

    Writes test and train loss/accuracy to TFEvents files.

    Args:
    name: (str) name of the model
    timestamp: (str) timestamp
    loss: (tf.op) loss operation
    accuracy: (tf.op) accuracy operation

    Return:
    writer_op: (dict) dictionary of the defined summary operations"""

    # Summary operations
    print('Writing to {}\n'.format(out_dir))

    summary_dir = os.path.join(PATHS['save_dir'], '{}_{}'.format(name, timestamp), 'summaries')

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', accuracy)

    # Train summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(summary_dir, 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir)

    # Test summaries
    test_summary_op = tf.summary.merge([loss_summary, acc_summary])
    test_summary_dir = os.path.join(summary_dir, 'test')
    test_summary_writer = tf.summary.FileWriter(test_summary_dir)

    # dictionary holding the writers
    writer_ops = {
        'loss': loss_summary,
        'accuracy': acc_summary,
        'train_op': train_summary_op,
        'test_op': test_summary_op,
        'train': train_summary_writer,
        'test': test_summary_writer,
    }

    return writer_ops


def summary_ops_task2(out_dir, name, timestamp, loss, accuracy, pixel_prob, ground_truth):
    """ Summary operations (Task 2, 3).

    Write test and train loss/accuracy and ground_truth images together with predicted
    and images and probabilities over pixels"""


def save(out_dir, name, timestamp, sess, input_x, input_y, phase, logits, accuracy, correct_prediction, losses, loss):
    """ Save the model to disk.

    Using tf's internal saving mechanism, save the
    model to directory.

    Args:
    out_dir: (str) path to where to save model
    name: (str) name of the model
    timestamp: (str) timestamp
    sess: current session object
    x: (tf.placeholder) input
    y_: (tf.placeholder) output
    logits: (tf.operation) logits
    accuracy: (tf.operation) accuracy operation
    correct_prediction: (tf.operation)
    losses: (tf.operation) Cross-entropy losses operation
    loss: (tf.operation) Cross-entropy loss operation

    Returns:
    Pass"""

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    # Add relevant operations and tensors to collection
    tf.add_to_collection('input_x', input_x)
    tf.add_to_collection('input_y', input_y)
    tf.add_to_collection('phase', phase)

    tf.add_to_collection('logits', logits)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('correct_prediction', correct_prediction)
    tf.add_to_collection('losses', losses)
    tf.add_to_collection('loss', loss)

    save_dir = os.path.join(out_dir, 'checkpoint')
    # Tensorflow assumes this directory already exists so we need to create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, '{}_{}.ckpt'.format(name, timestamp))
    save_path = saver.save(sess, save_path)

    print("Model saved in file: {}".format(save_path))


def load(sess, name, best_model=False):
    """ Load the model into the graph.

    Args:
    sess: (tf.Session) current session
    name: (str) name of the model save file to load
    (without file ending, normally of the form {name}_{timestamp})
    best_model: (bool) if we should load the models from the
    best_models directory or the normal save directory

    Returns:
    ops: (dict) dictionary of the operations:
    x: (tf.placeholder) input
    y_: (tf.placeholder) output
    logits: (tf.operation) logits
    accuracy: (tf.operation) accuracy operation
    correct_prediction: (tf.operation)
    losses: (tf.operation) Cross-entropy losses operation
    loss: (tf.operation) Cross-entropy loss operation"""

    # If we are evaluating on the optimal models or temp models
    if best_model:
        model_dir = PATHS['best_dir']
    else:
        model_dir = PATHS['save_dir']

    saver = tf.train.import_meta_graph(os.path.join(model_dir, name, 'checkpoint', name + '.ckpt.meta'))
    saver.restore(sess, os.path.join(model_dir, name, 'checkpoint', name + '.ckpt'))

    # Recollect operations
    input_x = tf.get_collection('input_x')[0]
    input_y = tf.get_collection('input_y')[0]
    phase = tf.get_collection('phase')[0]

    logits = tf.get_collection('logits')[0]
    correct_prediction = tf.get_collection('correct_prediction')
    accuracy = tf.get_collection('accuracy')[0]
    losses = tf.get_collection('losses')[0]
    loss = tf.get_collection('loss')[0]

    ops = {'input_x': input_x,
           'input_y': input_y,
           'phase': phase,
           'logits': logits,
           'correct_prediction': correct_prediction,
           'accuracy': accuracy,
           'losses': losses,
           'loss': loss}

    return ops


def eval(name, data):
    """ Evaluate the saved model for accuracy and loss (Task 1)

    We load the saved model and get the relevant operations
    in order to evaluate the accuracy and cross-entropy for
    the full train and test set. Since the train and test set
    are reasonably big such that RAM can't hold and perform the
    whole thing at once, we accumulate the sum and the loss and
    then average it at the end.

    Args:
    name: (str) name of the saved file without any file endings
    data: (tf_utils.Data) data class that holds the mnist data sets
    """

    # get the train and the test set
    train_images, train_labels = data.get_train_data()
    test_images, test_labels = data.get_test_data()

    # We are in test phase
    PHASE = False

    # Need to start a session
    with tf.Session() as sess:

        # Load saved operations
        ops = load(sess, name, best_model=True)

        # Accumulate loss and accuracy over batches of 5000
        batch_size = 5000

        assert data.train_size % batch_size == 0, "train_size {} has to be divisible by batch_size {}".format(data.train_size, batch_size)
        assert data.test_size % batch_size == 0, "test_size {} has to be divisible by batch_size {}".format(data.test_size, batch_size)
        train_iter = data.train_size/batch_size
        test_iter = data.test_size/batch_size

        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0

        for i in range(train_iter):
            print i
            x_batch = train_images[i * batch_size: (i + 1) * batch_size, :]
            y_batch = train_labels[i * batch_size: (i + 1) * batch_size, :]

            temp_loss, temp_acc = sess.run(
                [ops['loss'], ops['accuracy']],
                feed_dict={ops['input_x']: x_batch,
                           ops['input_y']: y_batch,
                           ops['phase']: PHASE})

            print "temp_acc {}, temp_loss {}".format(temp_acc, temp_loss)

            train_loss += temp_loss
            train_acc += temp_acc
        for i in range(test_iter):
            print i
            x_batch = test_images[i * batch_size: (i + 1) * batch_size, :]
            y_batch = test_labels[i * batch_size: (i + 1) * batch_size, :]

            temp_loss, temp_acc = sess.run(
                [ops['loss'], ops['accuracy']],
                feed_dict={ops['input_x']: x_batch,
                           ops['input_y']: y_batch,
                           ops['phase']: PHASE})

            print "temp_acc {}, temp_loss {}".format(temp_acc, temp_loss)

            test_loss += temp_loss
            test_acc += temp_acc


    # need to average it properly: batch_size * iterations = size
    train_loss /= train_iter
    train_acc /= train_iter
    test_loss /= test_iter
    test_acc /= test_iter

    # print the statistics
    print ("Train Loss: {:.5f}, Accuracy {:.5f}\n"
           "Test Loss: {:.5f}, Accuracy {:.5f}").format(train_loss,
                                                        train_acc,
                                                        test_loss,
                                                        test_acc)

    # reset default graph so that it is not polluted with these
    # variables and operations
    tf.reset_default_graph()

    # make sure the data is not saved
    del data


def eval_task1():
    """ Evaluate all of the models in task 1"""

    # test save file, should have accuracy over 0.8 and loss under 0.8
    save_files = 'task1LSTM32st1-lr0.001_bs256_dc0.99_1488162137 task1LSTM64st1-lr0.001_bs256_dc0.99_1488186148 task1LSTM128st1-lr0.001_bs256_dc0.99_1488233955 task1LSTM32st3-lr0.001_bs256_dc0.99_1488345867 task1GRU32st1-lr0.001_bs256_dc0.99_1488395608 task1GRU64st1-lr0.001_bs256_dc0.99_1488419108 task1GRU128st1-lr0.001_bs256_dc0.99_1488464311 task1GRU32st3-lr0.001_bs256_dc0.99_1488052668'.split(' ')

    data = Data(0)

    for name in save_files:
        print "Evaluating {}".format(name)
        eval(name, data)


if __name__ == '__main__':
    eval_task1()
