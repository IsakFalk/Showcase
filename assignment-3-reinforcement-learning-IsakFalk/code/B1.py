"""
B1: Score and frame counts from the three games under a random policy,
evaluated on 100 episodes. Report both average and standard deviation.
"""

import argparse

import numpy as np
import gym

from SETTINGS import GAMMA, MAX_EP_LEN, ENV_A
from B import Pong, MsPacman, Boxing

def main(args):
    print('Starting B1')
    # Get command line argument
    game = args.game
    episodes = args.episodes

    if game.lower() == 'pong':
        env = Pong()
    elif game.lower() == 'pacman':
        env = MsPacman()
    elif game.lower() == 'boxing':
        env = Boxing()

    # lists
    frame_count = []
    disc_total_rew = []
    raw_score = []

    # Run algorithm
    for episode in range(args.episodes):
        state = env.reset()
        temp_reward = 0
        temp_raw_reward = 0
        t = 0
        while True:
            t += 1
            action = env.sample_action()
            state, reward, done, info = env.step(action)
            temp_reward += reward * GAMMA**t
            temp_raw_reward += reward
            if done:
                frame_count.append(t)
                raw_score.append(temp_raw_reward)
                disc_total_rew.append(temp_reward)
                break

    # Arrays of statistics
    frame_count = np.asarray(frame_count)
    disc_total_rew = np.asarray(disc_total_rew)
    raw_score = np.asarray(raw_score)

    print('Game: {}'.format(game.lower()))
    print('Frame count: Mean {:.4f}, std {:.4f}'.format(frame_count.mean(), frame_count.std()))
    print('Raw score: Mean {:.4f}, std {:.4f}'.format(raw_score.mean(), raw_score.std()))
    print('Discounted total reward: Mean {:.4f}, std {:.4f}'.format(disc_total_rew.mean(), disc_total_rew.std()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("-g",
                        "--game",
                        dest="game",
                        help="Specify the game to evaluate.",
                        default="pong",
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

