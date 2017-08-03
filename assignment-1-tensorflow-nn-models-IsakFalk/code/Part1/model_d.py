# Model 1d

import tensorflow as tf

import part1lib as p1l

# Optimal parameters
batch_size = 200
epochs = 53
learn_rate = 0.1


def run_model_d(batch_size, epochs, learn_rate, test_set):
    """Run the model, used to learn hyperparameters"""

    # Data set
    train_set = p1l.mnist.train

    # We define the input and output
    x, y_ = p1l.get_placeholders()

    # Reshape the input
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First layer, Conv 3x3x16 + maxpool 2x2
    W_conv1 = p1l.weight_variable([3, 3, 1, 16], 'W_conv1')
    b_conv1 = p1l.bias_variable([16], 'b_conv1')

    # Convolute and downsample the output image to 14x14
    h_conv1 = tf.nn.relu(p1l.conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = p1l.max_pool_2x2(h_conv1)

    # Second layer, conv 3x3x16 + maxpool 2x2
    W_conv2 = p1l.weight_variable([3, 3, 16, 16], 'W_conv2')
    b_conv2 = p1l.bias_variable([16], 'b_conv2')

    # Convolute and downsample the image to 7x7
    h_conv2 = tf.nn.relu(p1l.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = p1l.max_pool_2x2(h_conv2)

    # Flatten image
    flat = tf.reshape(h_pool2, [-1, 784])

    # Non-linear layer
    flat_nl = tf.nn.relu(flat)

    # Linear + softmax
    W_linear = p1l.weight_variable([7 * 7 * 16, 10], 'W_linear')
    b_linear = p1l.bias_variable([10], 'b_linear')
    y_linear = tf.matmul(flat_nl, W_linear) + b_linear

    # Various variables needed
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_linear))

    # Training step, use SGD
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

    # Get accuracy
    accuracy = p1l.accuracy_op(y_linear, y_)

    # Initialise the session
    sess = p1l.start_session()

    # Train the model
    train_acc, test_acc, epoch_acc, it = p1l.train_model(sess, x, y_, train_set, test_set, train_step, batch_size, epochs, accuracy)

    # final accuracy of the model
    final_accuracy = sess.run(accuracy, feed_dict={x: test_set.images, y_: test_set.labels})

    # Save the model to disk
    p1l.save_to_disk(sess, 'd', accuracy, x, y_, y_linear)

    return final_accuracy, train_acc, test_acc, it, epoch_acc, sess


if __name__ == "__main__":

    print "Running model d"

    test_set = p1l.mnist.test

    final_accuracy, train_acc, test_acc, it, epoch_acc, sess = run_model_d(batch_size, epochs, learn_rate, test_set)

    print "Final test error: ", 1 - final_accuracy

    # Save accuracy plots
    p1l.plot_convergence('error_1d', train_acc, test_acc, it)
