"""
Preprocessing routines
"""

import os
import sys

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from SETTINGS import PATHS

def get_mnist():
    """ Get the train, test and validation set of mnist

    Makes sure that the path is consistent no matter from
    where you import the data.

    Args:
    None

    Returns:
    mnist: (data)
    """

    return input_data.read_data_sets(PATHS["data_dir"], one_hot=True)

def binarize(images, threshold=0.1):
    """ Binarize the images where we use threshold

    Args:
    images: (np.ndarray) N x H*W array of images
    threshold=0.1: (float) threshold for the binarization

    Returns:
    bin_images: (np.ndarray) Nx H*W array of binarized images
    """

    return (images < threshold).astype(np.float32)
