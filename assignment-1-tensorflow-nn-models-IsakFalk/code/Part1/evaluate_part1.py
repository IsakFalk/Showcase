### Evaluate the saved models to check that the test error is the same after loading

import os

import tensorflow as tf

import part1lib as p1l

# Change directory to the saved models

train_set = p1l.mnist.train
test_set = p1l.mnist.test

# If we want to plot the confusion matrix, set this flag

confusion_matrix = True

print "\n=== Part 1 ===\n"

# Model a, we evaluate the test and train error

print "Model 1a\n"

# Load file, create session, evaluate train and test error.

rel_path = os.path.join('..', '..', 'save', 'tf', 'a')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./' + rel_path + '/model_1a.ckpt.meta')
    saver.restore(sess, './' + rel_path + '/model_1a.ckpt')

    accuracy = tf.get_collection('accuracy')[0]
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    y = tf.get_collection('y')[0]

    train_accuracy = sess.run(accuracy, feed_dict={x: train_set.images, y_: train_set.labels})
    test_accuracy = sess.run(accuracy, feed_dict={x: test_set.images, y_: test_set.labels})

    # lists of actual vs predicted outputs to be used in confusion arrays
    predicted_digit = sess.run(tf.argmax(y, 1), feed_dict={x: test_set.images, y_: test_set.labels})
    true_digit = sess.run(tf.argmax(y_, 1), feed_dict={x: test_set.images, y_: test_set.labels})

print "Train error: {:.5f}\nTest error: {:.5f}\n".format(1 - train_accuracy, 1 - test_accuracy)

if confusion_matrix:
    p1l.plot_confusion_matrix('confusion_matrix_1a', predicted_digit, true_digit)

tf.reset_default_graph()


print "Model 1b\n"

# Load file, create session, evaluate train and test error.

rel_path = os.path.join('..', '..', 'save', 'tf', 'b')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./' + rel_path + '/model_1b.ckpt.meta')
    saver.restore(sess, './' +  rel_path + '/model_1b.ckpt')

    accuracy = tf.get_collection('accuracy')[0]
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    y = tf.get_collection('y')[0]

    train_accuracy = sess.run(accuracy, feed_dict={x: train_set.images, y_: train_set.labels})
    test_accuracy = sess.run(accuracy, feed_dict={x: test_set.images, y_: test_set.labels})

    # lists of actual vs predicted outputs to be used in confusion arrays
    predicted_digit = sess.run(tf.argmax(y, 1), feed_dict={x: test_set.images, y_: test_set.labels})
    true_digit = sess.run(tf.argmax(y_, 1), feed_dict={x: test_set.images, y_: test_set.labels})

print "Train error: {:.5f}\nTest error: {:.5f}\n".format(1 - train_accuracy, 1 - test_accuracy)

if confusion_matrix:
    p1l.plot_confusion_matrix('confusion_matrix_1b', predicted_digit, true_digit)

tf.reset_default_graph()


print "Model 1c\n"

# Load file, create session, evaluate train and test error.

rel_path = os.path.join('..', '..', 'save', 'tf', 'c')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(rel_path + '/model_1c.ckpt.meta')
    saver.restore(sess, rel_path + '/model_1c.ckpt')

    accuracy = tf.get_collection('accuracy')[0]
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    y = tf.get_collection('y')[0]

    train_accuracy = sess.run(accuracy, feed_dict={x: train_set.images, y_: train_set.labels})
    test_accuracy = sess.run(accuracy, feed_dict={x: test_set.images, y_: test_set.labels})

    # lists of actual vs predicted outputs to be used in confusion arrays
    predicted_digit = sess.run(tf.argmax(y, 1), feed_dict={x: test_set.images, y_: test_set.labels})
    true_digit = sess.run(tf.argmax(y_, 1), feed_dict={x: test_set.images, y_: test_set.labels})

print "Train error: {:.5f}\nTest error: {:.5f}\n".format(1 - train_accuracy, 1 - test_accuracy)


if confusion_matrix:
    p1l.plot_confusion_matrix('confusion_matrix_1c', predicted_digit, true_digit)

tf.reset_default_graph()


print "Model 1d\n"

# Load file, create session, evaluate train and test error.

rel_path = os.path.join('..', '..', 'save', 'tf', 'd')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(rel_path + '/model_1d.ckpt.meta')
    saver.restore(sess, rel_path + '/model_1d.ckpt')

    accuracy = tf.get_collection('accuracy')[0]
    x = tf.get_collection('x')[0]
    y_ = tf.get_collection('y_')[0]
    y = tf.get_collection('y')[0]

    train_accuracy = sess.run(accuracy, feed_dict={x: train_set.images, y_: train_set.labels})
    test_accuracy = sess.run(accuracy, feed_dict={x: test_set.images, y_: test_set.labels})

    # lists of actual vs predicted outputs to be used in confusion arrays
    predicted_digit = sess.run(tf.argmax(y, 1), feed_dict={x: test_set.images, y_: test_set.labels})
    true_digit = sess.run(tf.argmax(y_, 1), feed_dict={x: test_set.images, y_: test_set.labels})

print "Train error: {:.5f}\nTest error: {:.5f}\n".format(1 - train_accuracy, 1 - test_accuracy)


if confusion_matrix:
    p1l.plot_confusion_matrix('confusion_matrix_1d', predicted_digit, true_digit)

tf.reset_default_graph()
