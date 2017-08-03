"""
Vizualisation functions to be used with TFEvents files.
"""

import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib as mlp
from matplotlib import pyplot as plt
import seaborn as sns

from SETTINGS import *

def confusion_matrix(name, log_dir):
    """ Calculate the confusion matrix for a run.

    Args:
    name: (str) name of the model to save
    log_dir: (str) name of the directory of the logs
    """

    confusion_mat = 0 # fill in
    ax = sns.heatmap(confusion_mat, annot=True, fmt="d")
    ax.savefig(os.path.join(PATHS['img_dir'], '{}_confusion_matrix.jpg'.format(name)))
    plt.clf()


def time_series():
    """ Plots and saves the time series of the loss and accuracy

    We pass the function a directory of the TFEvents logfiles which
    we will use to get numpy arrays of the loss and the accuracy over
    training and test data.

    Args:
    name: (str) name of the model to save
    log_dir: (str) name of the directory of the logs"""

    # Need to get the iter array, loss and accuracy for test and train
    it = 0
    test_acc = 0
    train_acc = 0
    test_loss = 0
    train_loss = 0

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1)

    ax_loss.set_title('Loss over iterations')
    ax_loss.plot(it, train_loss, color='r', it, test_loss, color='b')
    ax_loss.set_xlabel('iterations')
    ax_loss.set_ylabel('Cross-entropy loss')

    ax_acc.set_title('Accuracy over iterations')
    ax_acc.plot(it, train_acc, color='r', it, test_acc, color='b')
    ax_acc.set_xlabel('iterations')
    ax_acc.set_ylabel('Accuracy')

    ax.savefig(os.path.join(PATHS['img_dir'], '{}_time_series.jpg'.format(name)))
    plt.clf()


def visualize_image(out_dir, pixel_prob, input_x, num_images=3, random=False):
    """ Visualize the images from task 2.

    Saves an image where we compare the ground truth image
    (left) to the outputs (probabilities over pixels, right).

    Args:
    pixel_prob: (np.ndarray) N x 784 array of the pixel probabilites
    input_x; (np.ndarray) N x 784 ground truth image
    num_images: (int) number of images to plot (num_images rows, 2 cols)
    random: (bool) shuffle the images plotted if true

    Returns:"""

    assert pixel_prob.shape == input_x.shape, "The probabilites and ground truth must have the same shape, current prob shape: {}, input shape {}".format(str(pixel_prob.shape), str(input_x.shape))
    assert pixel_prob.shape[1] == 784, "The image has to be 784 pixels big, not {}".format(pixel_prob.shape[1])

    batch_size = pixel_prob.shape[0]

    # get random index if random is True, else first 3 images
    if random:
        index = np.random.choice(batch_size, num_images, replace=False)
    else:
        index = np.arange(3)

    
