"""
Task A3.
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
    epochs = args.epochs
    batch_size = args.batch_size

    # Create random batch, 2000 steps
    states, actions, rewards, next_states, dones = create_random_batch()

    # Create directory and get save path
    save_path = create_save_directory(name, learning_rate)

    # Environment
    env = CartPoleEnv()

    # Keep necessary statistics
    ep_len = []
    disc_tot_reward = []
    losses = []
    iter_epochs = []

    # Optimal disc. total reward
    opt_disc_tot_rew = -1

    with tf.Session() as sess:
        # Create agent

        # Batch data
        batch_dict = dict(states=states,
                          actions=actions,
                          rewards=rewards,
                          next_states=next_states,
                          dones=dones)

        if args.estimator.lower() == 'linear':
            estimator = Linear(sess, learning_rate, save_path)
        else:
            estimator = OfflineNeuralNet(sess, learning_rate, save_path)

        # create the agent
        agent = AgentA3(estimator, batch_size, batch_dict)

        # Train the model
        for epoch in range(epochs):
            # Train one epoch
            agent.train_estimator()
            # Evaluate agent
            if epoch % 1 == 0:
                # temporary placeholders
                mean_ep_len, mean_reward, _ = env.evaluate_performance(agent, num_episodes=20)
                mean_loss = agent.get_loss()
                ep_len.append(mean_ep_len)
                disc_tot_reward.append(mean_reward)
                losses.append(mean_loss)
                iter_epochs.append(epoch)
                print('Epoch: {} | Length: {:.2f} | Reward: {:.5f} | Loss: {:.5f}'.format(epoch,
                                                                                          mean_ep_len,
                                                                                          mean_reward,
                                                                                          mean_loss))
                # Write summaries of weights
                agent.write_summaries(epoch)
                # Only save model if mean discounted total reward increase
                if mean_reward > opt_disc_tot_rew:
                    opt_disc_tot_rew = mean_reward
                    agent.save_estimator()

        print('Training finished.')

        # Plot it
        plot_A(iter_epochs, ep_len, disc_tot_reward, losses, name, save_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-n",
                        "--name",
                        dest="name",
                        help="Specify the name of the model.",
                        default="A3",
                        type=str)
    parser.add_argument("-l",
                        "--learning_rate",
                        dest="learning_rate",
                        help="Specify the learning rate.",
                        default=0.01,
                        type=float)
    parser.add_argument("-es",
                        "--estimator",
                        dest="estimator",
                        help="Specify if we use a linear or neural net estimator.",
                        default="linear",
                        type=str)
    parser.add_argument("-b",
                        "--batch_size",
                        dest="batch_size",
                        help="Specify the batch_size.",
                        default=128,
                        type=int)
    parser.add_argument("-e",
                        "--epochs",
                        dest="epochs",
                        help="Specify the number of epochs to train the model for.",
                        default=2000,
                        type=int)

    args = parser.parse_args()

    # Run model
    main(args)
