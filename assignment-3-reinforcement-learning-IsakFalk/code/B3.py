"""
B2: Score and frame counts from the three games under an initilized but
not trained network, evaluated on 100 episodes.
Report both average and standard deviation.
"""

import argparse
import time
import os

import numpy as np
import tensorflow as tf
import matplotlib as mlp
#mlp.use('Agg')
from matplotlib import pyplot as plt
#import seaborn as sns
import gym

from SETTINGS import GAMMA, MAX_EP_LEN, ENV_A
from B import *

def main(args):
    print('Starting B3')
    # Get command line argument
    game = args.game
    episodes = args.episodes
    name = args.name
    learning_rate = args.learning_rate
    eval_step = args.eval_step

    if game.lower() == 'pong':
        env = Pong()
        env_eval = Pong()
    elif game.lower() == 'pacman':
        env = MsPacman()
        env_eval = MsPacman()
    elif game.lower() == 'boxing':
        env = Boxing()
        env_eval = Boxing()

    # flag
    done_training = False

    # lists
    frame_count = []
    disc_total_rew = []
    raw_score = []
    residuals = []
    losses = []

    # For checking saving
    opt_disc_tot_rew = -np.inf

    # save path
    save_path = create_save_directory(name, learning_rate)

    # Image processor
    process = ProcessImages()

    with tf.Session() as sess:
        # Make agent
        agent = CNNAgent(sess,
                         save_path=save_path,
                         n_flatten=(28//4)*(28//4)*32,
                         n_output=env.action_space)

        # keep track of
        steps = 0
        # Run algorithm
        for episode in range(args.episodes):
            # If we are done training break output
            if done_training:
                break

            print('Episode {}'.format(episode))
            temp_reward = 0
            temp_raw_reward = 0
            # buffers
            temp_states_buffer = np.zeros([1, 28, 28, 4], np.uint8)
            temp_action_buffer = 0
            temp_reward_buffer = 0
            temp_next_states_buffer = np.zeros([1, 28, 28, 4], np.uint8)

            state = process.process(sess, env.reset())
            # Stack the initial frame 4 times
            temp_states_buffer[0, :, :, :] = np.tile(state, (1, 1, 4))
            temp_next_states_buffer[0, :, :, :] = np.tile(state, (1, 1, 4))
            # train
            while True and not done_training:
                steps += 1
                action = agent.eps_greedy_action(temp_states_buffer)
                # Get output
                next_state, reward, done, info = env.step(action)
                # Process frames into correct form
                next_state = process.process(sess, next_state)
                # Roll the 4 stacked buffer
                temp_next_states_buffer = np.roll(temp_next_states_buffer, -1, axis=-1)
                temp_next_states_buffer[0, :, :, -1] = np.squeeze(next_state)
                # Train network
                agent.train_network(state, action, reward, next_state, done)
                state = next_state
                # Roll the 4-stacked buffer
                temp_states_buffer = np.roll(temp_states_buffer, -1, axis=-1)
                temp_states_buffer[0, :, :, -1] = np.squeeze(state)
                # evaluate the agent
                if done_training or steps > 1000000:
                    done_training = True
                if steps % eval_step == 0:
                    temp_frame_count, temp_reward, temp_score, temp_loss, temp_residual = eval_agent(sess, agent, env_eval, process)
                    frame_count.append(temp_frame_count)
                    raw_score.append(temp_score)
                    disc_total_rew.append(temp_reward)
                    residuals.append(temp_residual)
                    losses.append(temp_loss)
                    print('Step: {:6.4f} | Score {:6.5f} | Discounted Reward {:6.5f} | Residual {:6.5f} | Loss {:6.5f}'.format(steps,
                                                                                                                               float(temp_score),
                                                                                                                               float(temp_reward),
                                                                                                                               float(temp_residual),
                                                                                                                               float(temp_loss)))
                    if temp_reward > opt_disc_tot_rew:
                        opt_disc_tot_rew = temp_reward
                        agent.save_estimator()
                if done or done_training:
                    break

    # Arrays of statistics
    frame_count = np.asarray(frame_count)
    disc_total_rew = np.asarray(disc_total_rew)
    raw_score = np.asarray(raw_score)
    residuals = np.asarray(residuals)
    losses = np.asarray(losses)

    # Save numpy arrays
    np.save(os.path.join(save_path, 'frame_count.npy'), frame_count)
    np.save(os.path.join(save_path, 'disc_total_rew.npy'), disc_total_rew)
    np.save(os.path.join(save_path, 'raw_score.npy'), raw_score)
    np.save(os.path.join(save_path, 'residuals.npy'), residuals)
    np.save(os.path.join(save_path, 'losses.npy'), losses)
    print('Saved all arrays to file.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-g",
                        "--game",
                        dest="game",
                        help="Specify the game to evaluate.",
                        default="pong",
                        type=str)
    parser.add_argument("-n",
                        "--name",
                        dest="name",
                        help="Specify the name of the save file.",
                        default="B",
                        type=str)
    parser.add_argument("-l",
                        "--learning_rate",
                        dest="learning_rate",
                        help="Specify the learning rate",
                        default=0.00001,
                        type=float)
    parser.add_argument("-v",
                        "--eval_step",
                        dest="eval_step",
                        help="Specify on what iterations to get evaluation.",
                        default=50000,
                        type=int)
    parser.add_argument("-e",
                        "--episodes",
                        dest="episodes",
                        help="Specify the number of episodes",
                        default=100,
                        type=int)
    args = parser.parse_args()

    # Run model
    main(args)
