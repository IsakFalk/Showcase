"""
Task A7.
Environment parameters are set to be:
Environment: 'CartPole-v0'
Reward: 0 on non-terminating steps, -1 on termination
Max episode length: 300
Discount factor: 0.99
"""

import argparse
import time
import os

import numpy as np
import tensorflow as tf
import matplotlib as mlp
mlp.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import gym

from A_utils import *
from SETTINGS import GAMMA, MAX_EP_LEN, ENV_A, PATHS

# Constants
NUM_EP = 2000

def main(args):
    # Command line arguments
    name = args.name
    learning_rate = args.learning_rate

    # Create directory and get save path
    save_path = create_save_directory(name, learning_rate)

    # Environment
    env = CartPoleEnv()

    # Keep necessary statistics
    # 100 runs and we get statistics every 20th episode
    runs = args.runs
    iter_get = 20
    num_iter = 2000//iter_get
    ep_len = np.zeros([runs, num_iter])
    disc_tot_reward = np.zeros([runs, num_iter])
    residuals = np.zeros([runs, num_iter])
    losses = np.zeros([runs, num_iter])
    iter_episodes = np.arange(0, 2000, iter_get)

    # Optimal disc. total reward
    opt_disc_tot_rew = -1

    for run in range(runs):
        tf.reset_default_graph()
        with tf.Session() as sess:
            # Create estimator
            estimator = SARSA(sess,
                              save_path,
                              learning_rate,
                              hidden_units=args.units,
                              input_size=4,
                              output_size=2)
            # create the agent
            agent = OnlineAgent(estimator, eps=0.05)

            # Timers
            t1 = time.time()

            for episode in range(NUM_EP):
                # Train one episode
                env.run_update_episode(agent)
                # Evaluate agent
                if episode % iter_get == 0:
                    # temporary placeholders
                    mean_ep_len, mean_reward, residue = env.evaluate_performance(agent, num_episodes=20, count=episode)
                    ep_len[run, episode//iter_get] = mean_ep_len
                    disc_tot_reward[run, episode//iter_get] = mean_reward
                    residuals[run, episode//iter_get] = np.mean(residue)
                    losses[run, episode//iter_get] = np.mean(residue ** 2)
                    t2 = time.time()
                    time_taken = t2 - t1
                    t1 = t2
                    print('run: {} | episode: {:4} | Length: {:6.2f} | Reward: {:4.5f} | Loss: {:4.5f} | Time taken: {:6.2f}'.format(run,
                                                                                                                                     episode,
                                                                                                                                     mean_ep_len,
                                                                                                                                     mean_reward,
                                                                                                                                     np.mean(residue ** 2),
                                                                                                                                     time_taken))
                    # Only save model if mean discounted total reward increase,
                    # and if on first run
                    if mean_reward > opt_disc_tot_rew and run == 0:
                        opt_disc_tot_rew = mean_reward
                        agent.save_estimator()

    # Plot it
    plot_A_2(iter_episodes, ep_len, disc_tot_reward, residuals, losses, name, save_path)

    # Save numpy arrays
    np.save(os.path.join(save_path, 'ep_len.npy'), ep_len)
    np.save(os.path.join(save_path, 'disc_tot_reward.npy'), disc_tot_reward)
    np.save(os.path.join(save_path, 'residuals.npy'), residuals)
    np.save(os.path.join(save_path, 'losses.npy'), losses)
    np.save(os.path.join(save_path, 'iter_episodes.npy'), iter_episodes)
    print('Saved all arrays to file.')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-n",
                        "--name",
                        dest="name",
                        help="Specify the name of the model.",
                        default="A6",
                        type=str)
    parser.add_argument("-l",
                        "--learning_rate",
                        dest="learning_rate",
                        help="Specify the learning rate.",
                        default=0.00001,
                        type=float)
    parser.add_argument("-u",
                        "--units",
                        dest="units",
                        help="Specify the number of hidden units.",
                        default=100,
                        type=int)
    parser.add_argument("-r",
                        "--runs",
                        dest="runs",
                        help="Specify the number of runs to average over.",
                        default=100,
                        type=int)

    args = parser.parse_args()

    # Run model
    main(args)

