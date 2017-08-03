### Model 1b

import part1lib as p1l

import tensorflow as tf

### Optimal parameters
batch_size = 200
epochs = 83
learn_rate = 0.1

def run_model_b(batch_size, epochs, learn_rate, test_set):
    """Run the model, used to learn hyperparameters"""

    # Data set
    train_set  = p1l.mnist.train

    # We define the input and output
    x, y_ = p1l.get_placeholders()

    # And the weights and biases which we will train for each layer

    # Layer 1 (ReLU)
    W1 = p1l.weight_variable([784, 128], 'W1')
    b1 = p1l.bias_variable([1, 128], 'b1')

    # Intermediate output vectors for the layer 1
    layer1 = tf.matmul(x, W1) + b1
    layer1 = tf.nn.relu(layer1)

    # Layer 2 (linear)
    W2 = p1l.weight_variable([128, 10], 'W2')
    b2 = p1l.bias_variable([1, 10], 'b2')

    # Output vector
    y = tf.matmul(layer1,W2) + b2

    # Cost function is cross-entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Training step, use SGD
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)

    # Get accuracy
    accuracy = p1l.accuracy_op(y, y_)

    # Initialise the session
    sess = p1l.start_session()

    # Train the model
    train_acc, test_acc, epoch_acc, it = p1l.train_model(sess, x, y_, train_set, test_set, train_step, batch_size, epochs, accuracy)

    # final accuracy of the model
    final_accuracy = sess.run(accuracy, feed_dict={x: test_set.images, y_: test_set.labels})

    # Save the model to disk
    p1l.save_to_disk(sess, 'b', accuracy, x, y_, y)

    return final_accuracy, train_acc, test_acc, it, epoch_acc, sess


if __name__ == "__main__":

    print "Running model b"

    test_set = p1l.mnist.test

    final_accuracy, train_acc, test_acc, it, epoch_acc, sess = run_model_b(batch_size, epochs, learn_rate, test_set)

    print "Final test error: ", 1 - final_accuracy

    # Save accuracy plots
    p1l.plot_convergence('error_1b', train_acc, test_acc, it)
