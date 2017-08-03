"""
Task A1.
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
NUM_EP = 3

# Instantiate environment
env = CartPoleEnv()

# Statistics
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
            print("Episode finished after {} timesteps".format(t+1))
            ep_len.append(t+1)
            disc_total_rew.append(temp_reward)
            break


print('Episode lengths: {}, {}, {}'.format(*ep_len))
round_disc_tot_rew = [round(x, 3) for x in disc_total_rew]
print('Discounted total rewards: {}, {}, {}'.format(*round_disc_tot_rew))
