"""
Task 2. Specify the number of stacks and other things from the command line
"""

import argparse
import time
import datetime
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from SETTINGS import *

import preprocess
import tf_utils2 as tf_utils


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
        seq_length = tf.placeholder(tf.int32, name='sequence_length')
        input_x = tf.placeholder(tf.float32, [None, None], name='input_x')

    # Putting all preprocessing steps on the CPU to improve speed
    with tf.device('/cpu:0'):
        reshaped_x = tf.expand_dims(input_x, axis=2)
        # need the shifted labels for the loss function
        shifted_x = input_x[:, 1:]

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
                                           dtype=tf.float32,
                                           sequence_length=seq_length)

        with tf.name_scope('logistic_regression'):
            W = tf.Variable(tf.truncated_normal([n_hidden, 1], stddev=2.0/np.sqrt(n_hidden)), name='W')
            b = tf.Variable(tf.zeros([1]), name='b')

            # Function to do batch matrix multiplication
            def batch_matmul(x):
                return tf.add(tf.matmul(x, W), b)
            logits = tf.map_fn(batch_matmul, outputs)

            logits = tf.reshape(logits, shape=[-1, 784], name='logits')
            # Need to remove final pixel since we can only predict from the second
            # pixel to the last
            logits = logits[:, :-1]
            # probability of each pixel using input
            # Dimensions: [batch_size, time_steps (pixels: 784)]
            # logits[i,x] is the probability of pixel x in image i being filled in
            pixel_probs = tf.sigmoid(logits, name='pixel_probs')

    # Write weight summaries to file
    W_hist = tf.summary.histogram('W', W)
    b_hist = tf.summary.histogram('b', b)
    logits_hist = tf.summary.histogram('logits', logits)
    outputs_hist = tf.summary.histogram('outputs_RNN', outputs)
    weight_summary = tf.summary.merge([outputs_hist, W_hist, b_hist, logits_hist])
    weight_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, 'summaries', 'weights'))

    # check that softmax_cross_entropy_with_logits does the operation along the correct dimension
    # in this case.
    with tf.name_scope('cross_entropy'):
        # have to compute the cross entropy by hand since
        # tensorflow functions expect tensors of other forms than
        # we have
        losses = tf.reduce_mean(- (shifted_x * tf.log(pixel_probs) + (1 - shifted_x) * tf.log(1 - pixel_probs)), axis=1)
        loss = tf.reduce_mean(losses, name='loss')

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.cast(tf.round(pixel_probs), tf.int8), tf.cast(shifted_x, tf.int8))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    with tf.name_scope('initialize'):
        init = tf.global_variables_initializer()

    if DEBUG:
        log_device_placement = True
    else:
        log_device_placement = False

    with tf.name_scope('summaries'):
        # summaries
        writer_ops = tf_utils.summary_ops(out_dir, name, timestamp, loss, accuracy, pixel_probs, input_x)

    # Gather all of the operations in dictionary that we can pass to save function
    ops = {
        'input_x': input_x,
        'seq_length': seq_length,
        'logits': logits,
        'pixel_probs': pixel_probs,
        'losses': losses,
        'loss': loss,
        'accuracy': accuracy,
    }

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
            tf_utils.debug(sess, input_x, outputs, W, b, logits, losses, loss, correct_prediction, accuracy)

        # TRAIN MODEL

        # evaluation size
        EVAL_SIZE = 800
        # we are in training phase (only need this if doing BN)
        #PHASE = True
        # sequence length is 784
        sl = 784
        # Keep track of best test accuracy so far and use it to save
        best_test_acc = 0
        # number of iterations in one epoch
        iter_epochs = data.train_size/batch_size

        for epoch in range(epochs):
            for i in tqdm(range(iter_epochs), disable=(not PROGRESS)):
                x_batch, _ = data.get_train_batch()
                feed_dict = {
                    input_x: x_batch,
                    seq_length: sl
                    #phase: PHASE
                }
                _ = sess.run(
                    train_op,
                    feed_dict=feed_dict)

            # Save summary statistics after each epoch

            # train
            x_batch, _ = data.get_random_train_batch(EVAL_SIZE)
            feed_dict = {
                input_x: x_batch,
                seq_length: sl
            }
            train_summaries, train_loss, train_acc = sess.run(
                [writer_ops['train_op'], loss, accuracy],
                feed_dict=feed_dict)
            writer_ops['train'].add_summary(train_summaries, epoch * iter_epochs)

            # test
            x_batch, _ = data.get_random_test_batch(EVAL_SIZE)
            feed_dict = {
                input_x: x_batch,
                seq_length: sl
            }
            test_summaries, test_loss, test_acc = sess.run(
                [writer_ops['test_op'], loss, accuracy],
                feed_dict=feed_dict)
            writer_ops['test'].add_summary(test_summaries, epoch * iter_epochs)

            # weight statistics
            weight_summaries = sess.run(weight_summary,
                                        feed_dict={
                                            input_x: x_batch,
                                            seq_length: sl
                                        })
            weight_summary_writer.add_summary(weight_summaries, i)

            print "Epoch {}. Training: Loss {:.4f}, Accuracy {:.4f}. Test: Loss {:.4f}, Accuracy {:.4f}".format(epoch, train_loss, train_acc, test_loss, test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # save the model
                tf_utils.save(out_dir, name, timestamp, sess, ops)


        # check output
        if DEBUG:
            # get just 5 random examples from the train set to test on
            x, _ =data.get_random_train_batch(5)
            input_x, outputs, logits, losses, loss, correct_prediction, accuracy = sess.run(
                [input_x, input_y, output, logits, losses, loss, correct_prediction, accuracy],
                feed_dict={input_x: x, phase: True})

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
                        default="TEST",
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
                        default=50,
                        type=int)
    parser.add_argument("-l",
                        "--learning_rate",
                        dest="learning_rate",
                        help="learning rate of the model.",
                        default=0.001,
                        type=float)
    parser.add_argument("-b",
                        "--batch_size",
                        dest="batch_size",
                        help="batch size we use in the current model.",
                        default=256,
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
