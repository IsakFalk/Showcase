"""
Task 1. Specify the number of stacks and other things from the command line
"""

import argparse
import time
import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from SETTINGS import *

import preprocess
import tf_utils1 as tf_utils


def main(args):
    # Parameters
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    stacks = args.stacks
    epochs = args.epochs
    cell_type = args.cell_type
    decay = args.decay
    n_hidden = args.hidden_units
    n_input = N_INPUT
    n_classes = N_CLASSES
    n_steps = N_STEPS

    # Flags
    if args.debug.lower() == 'yes':
        DEBUG = True
    else:
        DEBUG = False
    if args.progress.lower() == 'yes':
        PROGRESS = True
    else:
        PROGRESS = False

    # Make name reflect hyperparameter choices
    name = '{n}{cell}{units}st{stacks}-lr{lr}_bs{bs}_dc{dc}'.format(n=args.name,
                                                                    cell=cell_type,
                                                                    units=n_hidden,
                                                                    stacks=stacks,
                                                                    lr=learning_rate,
                                                                    bs=batch_size,
                                                                    dc=decay)
    timestamp = str(int(time.time()))
    out_dir = os.path.join(PATHS['save_dir'], '{}_{}'.format(name, timestamp))

    # variables
    data = tf_utils.Data(batch_size)

    with tf.name_scope('input'):
        input_x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        input_y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        keep_prob = tf.placeholder(tf.float32, name='keep_probability')
        phase = tf.placeholder(tf.bool, name='phase')

    # Putting all preprocessing steps on the CPU to improve speed
    with tf.device('/cpu:0'):
        # for testing that it works we can change number of pixels to feed
        # in at each timestep
        times_pixels = 1
        reshaped_x = tf.reshape(input_x, shape=[-1, 784/times_pixels, times_pixels])

    # RNN cell
    with tf.variable_scope('Network'):
        # Capitalize string
        cell_type = cell_type.upper()

        # We define the cell that goes into the dynamic rnn
        with tf.name_scope('cell'):
            # Define a lstm cell with tensorflow
            if cell_type == 'LSTM':
                RNN_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
            elif cell_type == 'GRU':
                RNN_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
            else:
                print 'Wrong cell_type ({}) provided. Exiting...'.format(cell_type)
                exit()

        if stacks > 1:
            RNN_cell = tf.nn.rnn_cell.MultiRNNCell([RNN_cell] * stacks)

        outputs, state = tf.nn.dynamic_rnn(cell=RNN_cell,
                                           inputs=reshaped_x,
                                           dtype=tf.float32)

        # Need to transpose due to how dynamic_rnn returns values
        outputs = tf.transpose(outputs, [1, 0, 2])
        # Only care about the final output
        output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
        # Batch normalisation
        output_BN = tf_utils.batch_norm_wrapper(output, phase, decay=decay)

        with tf.variable_scope('output'):
            # Middle layer is 100 units
            M = 100

            with tf.variable_scope('layer_1'):
                W1 = tf.Variable(tf.truncated_normal([n_hidden, M], stddev=2.0/np.sqrt(n_hidden)), name='W1')
                b1 = tf.Variable(tf.zeros([M]), name='b1')
                layer_1 = tf.add(tf.matmul(output_BN, W1), b1)
                layer_1 = tf.nn.relu(layer_1)
                # Batch normalisation
                layer_1_BN = tf_utils.batch_norm_wrapper(layer_1, phase, decay=decay)

            with tf.variable_scope('layer_2'):
                W2 = tf.Variable(tf.truncated_normal([M, n_classes], stddev=2.0/np.sqrt(M)), name='W2')
                b2 = tf.Variable(tf.zeros([n_classes]), name='b2')
                logits = tf.add(tf.matmul(layer_1_BN, W2), b2, name='logits')

            W1_hist = tf.summary.histogram('W1', W1)
            b1_hist = tf.summary.histogram('b1', b1)
            W2_hist = tf.summary.histogram('W2', W2)
            b2_hist = tf.summary.histogram('b2', b2)
            logits_hist = tf.summary.histogram('logits', logits)
            outputs_hist = tf.summary.histogram('outputs_RNN', outputs)
            weight_summary = tf.summary.merge([outputs_hist, W1_hist, b1_hist, W2_hist, b2_hist, logits_hist])
            weight_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, 'summaries', 'weights'))

    with tf.name_scope('cross_entropy'):
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits)
        loss = tf.reduce_mean(losses, name='loss')

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    with tf.name_scope('initialize'):
        init = tf.global_variables_initializer()

    if DEBUG:
        log_device_placement = True
    else:
        log_device_placement = False

    with tf.name_scope('summaries'):
        # summaries
        writer_ops = tf_utils.summary_ops(out_dir, name, timestamp, loss, accuracy)

    # config for the session
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=log_device_placement)

    # Session
    with tf.Session(config=session_conf) as sess:

        # Write out summary of graph
        graph_summary = tf.summary.FileWriter(os.path.join(out_dir, 'summaries', 'graph'), graph=sess.graph)

        sess.run(init)

        if DEBUG:
            tf_utils.debug(sess, input_x, input_y, outputs, output, W1, b1, layer_1, W2, b2, logits, losses, loss, correct_prediction, accuracy)

        # TRAIN MODEL

        # evaluation size
        EVAL_SIZE = 800
        # we are in training phase
        PHASE = True

        # Keep track of best test accuracy so far and use it to save
        best_test_acc = 0

        # number of iterations in one epoch
        iter_epochs = data.train_size/batch_size

        for epoch in range(epochs):
            for i in tqdm(range(iter_epochs), disable=(not PROGRESS)):
                x_batch, y_batch = data.get_train_batch()
                feed_dict = {
                    input_x: x_batch,
                    input_y: y_batch,
                    keep_prob: 1.0,
                    phase: PHASE
                }
                _ = sess.run(
                    train_op,
                    feed_dict=feed_dict)

            # Save summary statistics after each epoch

            # train
            x_batch, y_batch = data.get_random_train_batch(EVAL_SIZE)
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                keep_prob: 1.0,
                phase: PHASE
            }
            train_summaries, train_loss, train_acc = sess.run(
                [writer_ops['train_op'], loss, accuracy],
                feed_dict=feed_dict)
            writer_ops['train'].add_summary(train_summaries, epoch * iter_epochs)

            # test
            x_batch, y_batch = data.get_random_test_batch(EVAL_SIZE)
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                keep_prob: 1.0,
                phase: PHASE
            }
            test_summaries, test_loss, test_acc = sess.run(
                [writer_ops['test_op'], loss, accuracy],
                feed_dict=feed_dict)
            writer_ops['test'].add_summary(test_summaries, epoch * iter_epochs)

            # weight statistics
            weight_summaries = sess.run(weight_summary,
                                        feed_dict={
                                            input_x: x_batch,
                                            input_y: y_batch,
                                            keep_prob: 1.0,
                                            phase: PHASE
                                        })
            weight_summary_writer.add_summary(weight_summaries, i)

            print "Epoch {}. Training: Loss {:.4f}, Accuracy {:.4f}. Test: Loss {:.4f}, Accuracy {:.4f}".format(epoch, train_loss, train_acc, test_loss, test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # save the model
                tf_utils.save(out_dir,
                              name,
                              timestamp,
                              sess,
                              input_x,
                              input_y,
                              phase,
                              logits,
                              accuracy,
                              correct_prediction,
                              losses,
                              loss)

        # check output
        if DEBUG:
            # get just 5 random examples from the train set to test on
            x, y =data.get_random_train_batch(5)
            input_x, input_y, output, logits, losses, loss, correct_prediction, accuracy = sess.run(
                [input_x, input_y, output, logits, losses, loss, correct_prediction, accuracy],
                feed_dict={input_x:x, input_y:y, keep_prob: 1.0})

            print "input_x, shape {}\n{}\n".format(input_x.shape, input_x)
            print "input_y, shape {}\n{}\n".format(input_y.shape, input_y)
            print "output, shape {}\n{}\n".format(output.shape, output)
            print "logits, shape {}\n{}\n".format(logits.shape, logits)
            print "losses, shape {}\n{}\n".format(losses.shape, losses)
            print "loss, shape {}\n{}\n".format(loss.shape, loss)
            print "correct_prediction, shape {}\n{}\n".format(correct_prediction.shape, correct_prediction)
            print "accuracy, shape {}\n{}\n".format(accuracy.shape, accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # OS parameters
    parser.add_argument("-n",
                        "--name",
                        dest="name",
                        help="Specify the name of the model.",
                        default="Model1a",
                        type=str)
    parser.add_argument("-u",
                        "--hidden_units",
                        dest="hidden_units",
                        help="Number of hidden units in the hidden layer.",
                        default=32,
                        type=int)
    parser.add_argument("-c",
                        "--cell_type",
                        dest="cell_type",
                        help="Cell type in the RNN.",
                        default="LSTM",
                        type=str)
    # Hyperparameters
    parser.add_argument("-e",
                        "--epochs",
                        dest="epochs",
                        help="Number of epochs to train the model.",
                        default=10,
                        type=int)
    parser.add_argument("-l",
                        "--learning_rate",
                        dest="learning_rate",
                        help="learning rate of the model.",
                        default=1e-4,
                        type=float)
    parser.add_argument("-b",
                        "--batch_size",
                        dest="batch_size",
                        help="batch size we use in the current model.",
                        default=200,
                        type=int)
    parser.add_argument("-st",
                        "--stacks",
                        dest="stacks",
                        help="Number of stacks in the RNN",
                        default=1,
                        type=int)
    parser.add_argument("-d",
                        "--debug",
                        dest="debug",
                        help="Print out debugging information to the console.",
                        default="no",
                        type=str)
    parser.add_argument("-p",
                        "--progress",
                        dest="progress",
                        help="Use the progress line during training to show how far an epoch has come",
                        default="no",
                        type=str)
    parser.add_argument("-dc",
                        "--decay",
                        dest="decay",
                        help="Decay hyperparameter we use in the models",
                        default=0.99,
                        type=float)
    args = parser.parse_args()

    # Run model
    main(args)
