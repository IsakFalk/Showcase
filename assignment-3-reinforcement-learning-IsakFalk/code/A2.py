"""
Task A2.
Environment parameters are set to be:
Environment: 'CartPole-v0'
Reward: 0 on non-terminating steps, -1 on termination
Max episode length: 300
Discount factor: 0.99
"""

import numpy as np
import gym

from A_utils import CartPoleEnv
from SETTINGS import GAMMA, MAX_EP_LEN, ENV_A

# Constants
NUM_EP = 100

# Instantiate agent and env
env = CartPoleEnv()

# lists
ep_len = []
disc_total_rew = []

# Run algorithm
for _ in range(NUM_EP):
    temp_reward = 0
    state = env.reset()
    for t in range(MAX_EP_LEN):
        action = env.env.action_space.sample()
        state, reward, done, info = env.step(action)
        temp_reward += reward * GAMMA**t
        if done:
            ep_len.append(t+1)
            disc_total_rew.append(temp_reward)
            break

# Arrays of statistics
ep_len = np.asarray(ep_len)
disc_total_rew = np.asarray(disc_total_rew)

print('Episode length: Mean {}, std {}'.format(round(ep_len.mean(), 4), round(ep_len.std(), 4)))
print('Discounted total reward: Mean {}, std {}'.format(round(disc_total_rew.mean(), 4), round(disc_total_rew.std(), 4)))
