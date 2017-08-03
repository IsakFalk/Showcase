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

# Put path to the directory to plot here
save_path = 'None'

disc_tot_reward = np.load(os.path.join(save_path, 'disc_total_rew.npy'))
raw_score = np.load(os.path.join(save_path, 'raw_score.npy'))
residuals = np.load(os.path.join(save_path, 'residuals.npy'))
losses = np.load(os.path.join(save_path, 'losses.npy'))
eval_step = 1000
steps = np.arange(0, 100000, eval_step)

# Plot it
fig, axes = plt.subplots(2, 2)

# Axes 0
sns.tsplot(data=raw_score, time=steps, value='raw score', ax=axes[0, 0], color='red')
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_title('score')
# Axes 1
sns.tsplot(data=disc_tot_reward, time=steps, value='Discounted reward', ax=axes[0, 1], color='green')
axes[0, 1].set_xlabel('Reward')
axes[0, 1].set_title('reward')
# Axes 2
sns.tsplot(data=residuals, time=steps, value='residuals', ax=axes[1, 0], color='blue')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_title('Residuals')
# Axes 3
sns.tsplot(data=losses, time=steps, value='losses', ax=axes[1, 1], color='black')
axes[1, 1].set_xlabel('Reward')
axes[1, 1].set_title('Reward')

fig.tight_layout()
fig.savefig(os.path.join(save_path, save_path.split(os.sep)[-1] + '.png'))
print('Plot saved')
fig.clf()
