"""
Evaluate all of the saved models from A.
Evaluation is averaged reward and episode length of 100 episodes
of the CartPole environment
"""


import time
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib as mlp
from matplotlib import pyplot as plt
import seaborn as sns
import gym
import tensorflow as tf

from A_utils import *
from SETTINGS import GAMMA, MAX_EP_LEN, PATHS

import tf_utils

env = CartPoleEnv()

# A4
with tf.Session() as sess:
    print('A4')
    agent = LoadAgent(sess, 'A4-lr0.001_1490990734')
    print('Episode length {:.3f}, Total discounted reward {:.3f}'.format(*env.evaluate_performance(agent)))
tf.reset_default_graph()

# A5a
with tf.Session() as sess:
    print('A5a')
    agent = LoadAgent(sess, 'A5a-lr0.0001_1491055651')
    print('Episode length {:.3f}, Total discounted reward {:.3f}'.format(*env.evaluate_performance(agent)))
tf.reset_default_graph()

# A5b
with tf.Session() as sess:
    print('A5b')
    agent = LoadAgent(sess, 'A5b-lr1e-05_1491061724')
    print('Episode length {:.3f}, Total discounted reward {:.3f}'.format(*env.evaluate_performance(agent)))
tf.reset_default_graph()

# A6
with tf.Session() as sess:
    print('A6')
    agent = LoadAgent(sess, 'A6-lr0.0001_1491736882')
    print('Episode length {:.3f}, Total discounted reward {:.3f}'.format(*env.evaluate_performance(agent)))
tf.reset_default_graph()

# A7
with tf.Session() as sess:
    print('A7')
    agent = LoadAgent(sess, 'A7-lr0.0001_1491739194')
    print('Episode length {:.3f}, Total discounted reward {:.3f}'.format(*env.evaluate_performance(agent)))
tf.reset_default_graph()
