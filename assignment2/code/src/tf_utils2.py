"""
Utility functions that has to do with tensorflow for task 2.
Shares some functions with tf_utils1.py.
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


def summary_ops(out_dir, name, timestamp, loss, accuracy, pixel_probs, input_x):
    """ Summary operations (task 1).

    Writes test and train loss/accuracy to TFEvents files.

    Args:
    name: (str) name of the model
    timestamp: (str) timestamp
    loss: (tf.op) loss operation
    accuracy: (tf.op) accuracy operation
    pixel_probs: (tf.op) pixel probabilities
    input_x: (tf.op) ground truth image

    Return:
    writer_op: (dict) dictionary of the defined summary operations"""

    # Summary operations
    print('Writing to {}\n'.format(out_dir))

    summary_dir = os.path.join(PATHS['save_dir'], '{}_{}'.format(name, timestamp), 'summaries')

    # Fix images, we will have random batches so just pick first image in
    # each batch, shape needs to be [H, W, C], C=1 as we have greyscale
    NUM_IMAGES = 5 # number of images to save from each batch.

    # preprocessing of images to make it fit from 733 to 734 by padding
    pixel_probs_temp = pixel_probs[0:NUM_IMAGES, :]
    # Due to shift we will pad the pixel_probs with a zero in the front
    pixel_probs_pad = tf.concat(1, [tf.ones([NUM_IMAGES, 1]), pixel_probs_temp])
    pixel_reshaped = tf.reshape(pixel_probs_pad, shape=[NUM_IMAGES, 28, 28, 1])
    binarized_pixels = tf.round(pixel_reshaped)
    input_reshaped = tf.reshape(input_x[0:NUM_IMAGES, :], shape=[NUM_IMAGES, 28, 28, 1])

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', accuracy)
    pixel_summary = tf.summary.image('pixel_probs', pixel_reshaped)
    pred_summary = tf.summary.image('pixel_pred', binarized_pixels)
    ground_truth_summary = tf.summary.image('ground_truth', input_reshaped)

    # Train summaries
    train_summary_op = tf.summary.merge([loss_summary,
                                         acc_summary,
                                         pixel_summary,
                                         pred_summary,
                                         ground_truth_summary])
    train_summary_dir = os.path.join(summary_dir, 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir)

    # Test summaries
    test_summary_op = tf.summary.merge([loss_summary,
                                        acc_summary,
                                        pixel_summary,
                                        pred_summary,
                                        ground_truth_summary])
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


def save(out_dir, name, timestamp, sess, ops):
    """ Save the model to disk.

    Using tf's internal saving mechanism, save the
    model to directory.

    Args:
    out_dir: (str) path to where to save model
    name: (str) name of the model
    timestamp: (str) timestamp
    sess: current session object
    ops: (dict) dictionary containing all of the operations

    Returns:
    Pass"""

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    # Add all of the operations to the graph.
    # ops.iteritems iterates over the (key, value) pairs of the dict.
    # In this case, (key, value) corresponds to (name, opeeration)
    [tf.add_to_collection(op_name, operation) for op_name, operation in ops.iteritems()]

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
    logits: (tf.operation) logits
    accuracy: (tf.operation) accuracy operation
    pixel_probs: (tf.operation)
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
    seq_length = tf.get_collection('seq_length')[0]
    logits = tf.get_collection('logits')[0]
    pixel_probs = tf.get_collection('pixel_probs')[0]
    losses = tf.get_collection('losses')[0]
    loss = tf.get_collection('loss')[0]
    accuracy = tf.get_collection('accuracy')[0]

    ops = {
        'input_x': input_x,
        'seq_length': seq_length,
        'logits': logits,
        'pixel_probs': pixel_probs,
        'losses': losses,
        'loss': loss,
        'accuracy': accuracy,
    }

    return ops


def eval(name, data):
    """ Evaluate the saved model for accuracy and loss (Task 2a)

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

            temp_loss, temp_acc = sess.run(
                [ops['loss'], ops['accuracy']],
                feed_dict={ops['input_x']: x_batch,
                           ops['seq_length']: 784})

            print "temp_acc {}, temp_loss {}".format(temp_acc, temp_loss)

            train_loss += temp_loss
            train_acc += temp_acc
        for i in range(test_iter):
            print i
            x_batch = test_images[i * batch_size: (i + 1) * batch_size, :]

            temp_loss, temp_acc = sess.run(
                [ops['loss'], ops['accuracy']],
                feed_dict={ops['input_x']: x_batch,
                           ops['seq_length']: 784})

            print "temp_acc {}, temp_loss {}".format(temp_acc, temp_loss)

            test_loss += temp_loss
            test_acc += temp_acc

    # need to average it properly: batch_size * iterations = size, might have to divide by 784 as well
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


def eval_task2():
    """ Evaluate all of the models in task 2"""

    save_files = 'task2GRU32st1-lr0.001_bs256_dc0.99_1488480798 task2GRU64st1-lr0.001_bs256_dc0.99_1488488776 task2GRU128st1-lr0.001_bs256_dc0.99_1488496514 task2GRU32st3-lr0.001_bs256_dc0.99_1488504944'.split(' ')

    data = Data(0)

    for name in save_files:
        print "Evaluating {}".format(name)
        eval(name, data)


if __name__ == '__main__':
    eval_task2()

