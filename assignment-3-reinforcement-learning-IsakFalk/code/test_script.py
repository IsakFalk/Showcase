import argparse
import time
import os

import numpy as np
import tensorflow as tf
import matplotlib as mlp
from matplotlib import pyplot as plt
import seaborn as sns
import gym

from A_utils import *
from SETTINGS import GAMMA, MAX_EP_LEN, ENV_A, PATHS

save_path = '/home/isak/University/UCL_CSML/Modules/Advanced_methods_in_ML/Assignments/assignment-3-reinforcement-learning-IsakFalk/models/best_models/A4-lr0.001_1490990734'

episodes = np.load(save_path + '/iter_episodes.npy')
ep_len = np.load(save_path + '/ep_len.npy')
disc_tot_reward = np.load(save_path + '/disc_tot_reward.npy')
residuals = np.load(save_path + '/residuals.npy')
losses = np.load(save_path + '/losses.npy')

plot_A_2(episodes, ep_len, disc_tot_reward, residuals, losses, 'Test.png', './')
