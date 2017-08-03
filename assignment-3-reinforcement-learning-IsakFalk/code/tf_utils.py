"""
Utility functions that has to do with tensorflow
"""

import os
import sys

import numpy as np
import tensorflow as tf

from SETTINGS import *


def reset_graph(sess):
    """ Resets the graph in use. """

    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def save(sess, name):
    """ Save the graph"""
    saver = tf.train.Saver()
    save_path = os.path.join(PATHS['project_root'], name)
    save_path = saver.save(sess, save_path)
    print('Model saved in {}'.format(save_path))


def load(sess, name):
    """ Restore the graph from save file"""
    load_path = os.path.join(PATHS['best_dir'], name)
    saver = tf.train.Saver()
    saver.restore(sess, load_path)
    print('Model restored from {}'.format(load_path))
