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
    print('Starting B4')
    # Get command line argument
    game = args.game
    episodes = args.episodes

    # lists
    raw_score = []

    # Image processor
    process = ProcessImages()

    with tf.Session() as sess:
        # model directory
        models_dir = PATHS['best_dir']

        # load the relevant agent
        if game.lower() == 'pong':
            env = Pong()
            env_eval = Pong()
            agent = LoadAgent(sess, os.path.join(models_dir, 'pong-lr0.001_1491824892'))
        elif game.lower() == 'pacman':
            env = MsPacman()
            env_eval = MsPacman()
            agent = LoadAgent(sess, os.path.join(models_dir, 'pacman-lr0.001_1491832746'))
        elif game.lower() == 'boxing':
            env = Boxing()
            env_eval = Boxing()
            agent = LoadAgent(sess, os.path.join(models_dir, 'boxing-lr0.001_1491840929'))


        for episode in range(args.episodes):
            print('Episode {}'.format(episode))
            temp_raw_reward = 0
            temp_states_buffer = np.zeros([1, 28, 28, 4], np.uint8)
            state = process.process(sess, env.reset())
            # Stack the initial frame 4 times
            temp_states_buffer[0, :, :, :] = np.tile(state, (1, 1, 4))
            while True:
                action = agent.greedy_action(temp_states_buffer)
                # Get output
                state, reward, done, info = env.step(action)
                temp_raw_reward += reward
                # Process frames into correct form
                state = process.process(sess, state)
                # Attach new frame to state
                temp_states_buffer = np.roll(temp_states_buffer, -1, axis=-1)
                temp_states_buffer[0, :, :, -1] = np.squeeze(state)
                # evaluate the agent
                if done:
                    print('Score {}'.format(temp_raw_reward))
                    raw_score.append(temp_raw_reward)
                    break

    # make save directory
    save_path = os.path.join(PATHS['save_dir'], game + 'B4')
    os.makedirs(save_path, exist_ok=True)

    # Save numpy array
    raw_score = np.array(raw_score)
    print('Averaged Raw Score {:.2f}'.format(np.mean(raw_score)))
    np.save(os.path.join(save_path, 'raw_score.npy'), raw_score)
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
    parser.add_argument("-e",
                        "--episodes",
                        dest="episodes",
                        help="Specify the number of episodes",
                        default=100,
                        type=int)
    args = parser.parse_args()

    # Run model
    main(args)
