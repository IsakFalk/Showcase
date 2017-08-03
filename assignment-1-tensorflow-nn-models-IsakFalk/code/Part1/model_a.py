### Model 1a

import tensorflow as tf

import part1lib as p1l

### Optimal parameters
batch_size = 200
epochs = 85
learn_rate = 0.1


def run_model_a(batch_size, epochs, learn_rate, test_set):
    """Run the model, used to learn hyperparameters"""

    # Data set
    train_set  = p1l.mnist.train

    # We define the input and output
    x, y_ = p1l.get_placeholders()

    # And the weight and bias which we will train
    W = p1l.weight_variable([784, 10], 'W1')
    b = p1l.bias_variable([1, 10], 'b1')

    # Since we just have one linear layer followed by a softmax link
    y = tf.matmul(x, W) + b

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
    p1l.save_to_disk(sess, 'a', accuracy, x, y_, y)

    return final_accuracy, train_acc, test_acc, it, epoch_acc, sess


if __name__ == "__main__":

    print "Running model a"

    test_set = p1l.mnist.test

    final_accuracy, train_acc, test_acc, it, epoch_acc, sess = run_model_a(batch_size, epochs, learn_rate, test_set)

    print "Final test error: ", 1 - final_accuracy

    # Save accuracy plots
    p1l.plot_convergence('error_1a', train_acc, test_acc, it)
