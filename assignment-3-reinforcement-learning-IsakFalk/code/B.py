"""
Script for part B of the advanced assignment 3.
"""

import time
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib as mlp
#mlp.use('Agg')
from matplotlib import pyplot as plt
#import seaborn as sns
#sns.set_style({'lines.linewidth': 2.0})
import gym
import tensorflow as tf

from SETTINGS import GAMMA, MAX_EP_LEN, PATHS
import tf_utils


# Classes

class ProcessImages():
    """
    Process the raw atari images.

    Resize the images and convert them to grayscale and stack them in
    terms of 4. All of the images in the atari games takes the form of
    (210, 160, 3) RGB images.
    """
    def __init__(self, size=(28, 28)):
        self.input_frame = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.output = tf.image.rgb_to_grayscale(self.input_frame)
        # self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
        self.output = tf.image.resize_images(
            self.output, size=size, method=tf.image.ResizeMethod.BICUBIC)

    def process(self, sess, state):
        """
        Process an (210, 160, 3) RGB atari image into a
        (size[0], size[1], 1) grayscale image.
        """
        return sess.run(self.output, feed_dict={self.input_frame: state})


class Env():
    """ Base class for the Environments"""

    def __init__(self):
        self.env = gym.make('Pong-v3')
        self.action_space = 18

    def step(self, action):
        # action is one-hot vector, get number
        observation, reward, done, info =  self.env.step(action)
        reward = np.clip(reward, -1, 1)
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()

    def sample_action(self):
        return self.env.action_space.sample()


class Pong(Env):
    """ Class for the Pong-v3 environment"""

    def __init__(self):
        self.env = gym.make('Pong-v3')
        self.action_space = str(self.env.action_space).replace('(', ' ').replace(')', ' ').split()[-1]
        self.action_space = int(self.action_space)


class MsPacman(Env):
    """ Class for the MsPacman-v3 environment"""

    def __init__(self):
        self.env = gym.make('MsPacman-v3')
        self.action_space = str(self.env.action_space).replace('(', ' ').replace(')', ' ').split()[-1]
        self.action_space = int(self.action_space)


class Boxing(Env):
    """ Class for the Boxing environment"""

    def __init__(self):
        self.env = gym.make('Boxing-v3')
        self.action_space = str(self.env.action_space).replace('(', ' ').replace(')', ' ').split()[-1]
        self.action_space = int(self.action_space)


class LoadAgent():
    """ Load agents"""

    def __init__(self, sess, load_path):
        self.load_path = load_path
        self.sess = sess

        # get model
        model_dir = os.path.join(PATHS['best_dir'], load_path)
        # restore model
        saver = tf.train.import_meta_graph(os.path.join(model_dir, 'model.ckpt.meta'))
        saver.restore(sess, os.path.join(model_dir, 'model.ckpt'))

        # Get operations
        self.states = tf.get_collection('states')[0]
        self.greedy = tf.get_collection('greedy')[0]

    def greedy_action(self, state):
        action = self.sess.run(self.greedy, feed_dict={self.states: state})[0]
        return action


class CNNAgent():
    """ Agent of part B.

    Since all of the environments act the same
    we only need one type for all three of them"""

    def __init__(self,
                 sess,
                 save_path,
                 n_flatten=0,
                 n_output=6,
                 learning_rate=0.001,
                 buffer_size=100000,
                 batch_size=32,
                 copy_target=5000,
                 eps=0.1):
        self.sess = sess
        self.eps = eps
        self.batch_size = batch_size
        self.copy_target = copy_target
        self.n_output = n_output
        self.counter = 0
        self.save_path = save_path

        # Build network
        # Placeholders for the states of the things
        self.states = tf.placeholder(dtype=tf.uint8, shape=[None, 28, 28, 4], name='states')
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, n_output], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
        self.next_states = tf.placeholder(dtype=tf.uint8, shape=[None, 28, 28, 4], name='next_states')
        self.dones = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='dones')
        # Target network placeholders
        self.filter_1_target = tf.placeholder(dtype=tf.float32, shape=[6, 6, 4, 16], name='f_1_t')
        self.conv1_b_target = tf.placeholder(dtype=tf.float32, shape=[16], name='c_1_b_t')
        self.filter_2_target = tf.placeholder(dtype=tf.float32, shape=[4, 4, 16, 32], name='f_2_t')
        self.conv2_b_target = tf.placeholder(dtype=tf.float32, shape=[32], name='c_2_b_t')
        self.W1_target = tf.placeholder(dtype=tf.float32, shape=[n_flatten, 256], name='W_flat')
        self.b1_target = tf.placeholder(dtype=tf.float32, shape=[256], name='b_flat')
        self.W_output_target = tf.placeholder(dtype=tf.float32, shape=[256, n_output], name='W_output')
        self.b_output_target = tf.placeholder(dtype=tf.float32, shape=[n_output], name='b_output')

        # Replay buffer
        # actions are one-hot
        self.replay_buffer = dict(states=np.zeros((buffer_size, 28, 28, 4), dtype=np.uint8),
                                  actions=np.zeros((buffer_size, n_output), dtype=np.float32),
                                  rewards=np.zeros((buffer_size, 1), dtype=np.float32),
                                  next_states=np.zeros((buffer_size, 28, 28, 4), dtype=np.uint8),
                                  dones=np.zeros((buffer_size, 1), dtype=np.bool))
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.buffer_filled = False
        assert self.buffer_size >= self.batch_size, 'Need to have buffer_size >= batch_size'

        # Network
        # CNN: X (N, H, W, 4 (stacked frames)) -> X (N, 28, 28, 4) (greyscale)
        # -> filter (F: 6x6, S: 2, C: 16) + ReLU -> filter (F: 4x4, S: 2, C: 32) + ReLU
        # -> flatten -> Fully connected (U: 256) + ReLU -> Linear layer (256 -> |A|)
        # We use the following features and augmentations
        # Epsilon: 0.1
        # Discount factor: 0.99
        # Clip env rewards: [-1, 0, 1]
        # Batch size: 32
        # Optimizer: RMSprop (stepsize: 0.001)
        # Target network (switch every 5000 steps)
        # Experience replay buffer: Storage size 100000

        # Images are turned into greyscale and stacked in the environment class
        # Agent class get stacked images, one stack is (H, W, 4)

        # Network
        # First filter (width, height, n_in, n_out)
        self.filter_1 = tf.get_variable(name='filter_1',
                                  shape=[6, 6, 4, 16],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.conv1_b = tf.zeros([16], dtype=tf.float32)
        self.conv1 = tf.nn.conv2d(
            input=tf.to_float(self.states)/255.0,
            filter=self.filter_1,
            strides=[1, 2, 2, 1],
            padding="SAME")
        self.conv1 = tf.nn.relu(self.conv1 + self.conv1_b)
        # Target
        self.conv1_target = tf.nn.conv2d(
            input=tf.to_float(self.next_states)/255.0,
            filter=self.filter_1_target,
            strides=[1, 2, 2, 1],
            padding="SAME")
        self.conv1_target = tf.nn.relu(self.conv1_target + self.conv1_b_target)

        # Second filter (width, height, n_in, n_out)
        self.filter_2 = tf.get_variable(name='filter_2',
                                  shape=[4, 4, 16, 32],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.conv2_b = tf.zeros([32], dtype=tf.float32)
        self.conv2 = tf.nn.conv2d(
            input=self.conv1,
            filter=self.filter_2,
            strides=[1, 2, 2, 1],
            padding="SAME")
        self.conv2 = tf.nn.relu(self.conv2 + self.conv2_b)
        # Target
        self.conv2_target = tf.nn.conv2d(
            input=self.conv1_target,
            filter=self.filter_2_target,
            strides=[1, 2, 2, 1],
            padding="SAME")
        self.conv2_target = tf.nn.relu(self.conv2_target + self.conv2_b_target)

        # Flatten output
        self.flattened = tf.reshape(self.conv2, [-1, n_flatten])
        self.flattened_target = tf.reshape(self.conv2_target, [-1, n_flatten])

        # Fully connected
        self.W1 = tf.get_variable(name='W1',
                                  shape=[n_flatten, 256],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.b1 = tf.get_variable(name='b1',
                                  shape=[256],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(value=0.0))
        self.h = tf.nn.relu(tf.matmul(self.flattened, self.W1) + self.b1)
        self.h_target = tf.nn.relu(tf.matmul(self.flattened_target, self.W1_target) + self.b1_target)

        # Output over actions
        self.W_output = tf.get_variable(name='W_output',
                                        shape=[256, n_output],
                                        dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
        self.b_output = tf.get_variable(name='b_output',
                                        shape=[n_output],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(value=0.0))
        self.predictions = tf.nn.relu(tf.matmul(self.h, self.W_output) + self.b_output)
        self.predictions_target = tf.nn.relu(tf.matmul(self.h_target, self.W_output_target) + self.b_output_target)

        # Make mask using one hot vectors
        terminate = 1 - tf.cast(self.dones, tf.float32)
        self.Q_t = tf.reduce_sum(tf.multiply(self.predictions, self.actions), axis=1)
        self.Q_max = tf.reduce_max(self.predictions_target, axis=1)
        self.target = self.rewards + GAMMA * terminate * tf.stop_gradient(self.Q_max)

        # Squared loss
        self.delta = self.target - self.Q_t
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.delta), name='loss')
        self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

        # Greedy policy
        self.greedy = tf.argmax(self.predictions, axis=1)

        # Add everything to summary writers
        tf.summary.scalar('residual', tf.reduce_mean(self.delta))
        tf.summary.scalar('loss', self.loss)
        tf.summary.histogram('filter_1', self.filter_1)
        tf.summary.histogram('conv1_b', self.conv1_b)
        tf.summary.histogram('filter_2', self.filter_2)
        tf.summary.histogram('conv2_b', self.conv2_b)
        tf.summary.histogram('W_flatten', self.W1)
        tf.summary.histogram('b_flatten', self.b1)
        tf.summary.histogram('W_output', self.W_output)
        tf.summary.histogram('b_output', self.b_output)

        # Make summary dir
        os.makedirs(os.path.join(self.save_path, 'summaries'), exist_ok=True)

        # merge all summaries
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.save_path, 'summaries'))

        # Initializing the computational graph
        self.sess.run(tf.global_variables_initializer())

        # Copy variables to target network
        self.params = self.sess.run([self.filter_1,
                                     self.conv1_b,
                                     self.filter_2,
                                     self.conv2_b,
                                     self.W1,
                                     self.b1,
                                     self.W_output,
                                     self.b_output])

    def greedy_action(self, state):
        """ Perform a greedy policy selection

        state have the form of (1, 28, 28, 4) of 4 stacked frames
        of 28x28 pixels."""
        feed_dict={self.states: state}
        action = self.sess.run(self.greedy, feed_dict=feed_dict)
        return action[0]

    def eps_greedy_action(self, state):
        """ Perform epsilon greedy action"""
        random = np.random.binomial(1, self.eps)
        if random:
            action = np.random.choice(self.n_output, 1)
        else:
            action =  self.greedy_action(state)
        return action

    def get_loss(self, states, actions, rewards, next_states, dones):
        """ Get the loss and the residual"""
        actions = np.eye(self.n_output)[actions].reshape([-1, self.n_output])
        temp_params = self.sess.run([self.filter_1,
                                     self.conv1_b,
                                     self.filter_2,
                                     self.conv2_b,
                                     self.W1,
                                     self.b1,
                                     self.W_output,
                                     self.b_output])
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: np.asarray(rewards).reshape([-1, 1]),
                   self.next_states: next_states,
                   self.dones: np.asarray(dones).reshape([-1, 1]),
                   self.filter_1_target: temp_params[0],
                   self.conv1_b_target: temp_params[1],
                   self.filter_2_target: temp_params[2],
                   self.conv2_b_target: temp_params[3],
                   self.W1_target: temp_params[4],
                   self.b1_target: temp_params[5],
                   self.W_output_target: temp_params[6],
                   self.b_output_target: temp_params[7]}
        return self.sess.run([self.loss, self.delta], feed_dict=feed_dict)

    def update_target_network(self):
        """ Update the target network"""
        # Copy the network parameters
        self.params = self.sess.run([self.filter_1,
                                     self.conv1_b,
                                     self.filter_2,
                                     self.conv2_b,
                                     self.W1,
                                     self.b1,
                                     self.W_output,
                                     self.b_output])

    def populate_buffer(self, state, action, reward, next_state, done):
        """ Populate the buffer after the first 4 frames
        have been collected"""
        current_point = self.buffer_counter % self.buffer_size
        self.replay_buffer['states'][current_point, :, :, :] = state
        self.replay_buffer['actions'][current_point, :] = np.eye(self.n_output)[action]
        self.replay_buffer['rewards'][current_point, 0] = reward
        self.replay_buffer['next_states'][current_point, :, :, :] = next_state
        self.replay_buffer['dones'][current_point, 0] = done
        self.buffer_counter += 1

    def add_to_buffer(self, state, action, reward, next_state, done):
        """ Add the lists from the environment of the episode to the replay buffer"""
        # number of steps taken in previous episode
        start = self.buffer_counter % self.buffer_size
        prev = start - 1 % self.buffer_size
        # add to buffer
        self.replay_buffer['actions'][start, :] = np.eye(self.n_output)[[action]]
        self.replay_buffer['rewards'][start, :] = reward
        self.replay_buffer['dones'][start, 0] = done
        # Need to concatenate correctly
        # states
        prev_stacked_frames = self.replay_buffer['states'][prev, :, :, 1:]
        new_frame = np.concatenate((prev_stacked_frames, state), axis=-1)
        self.replay_buffer['states'][start, :, :, :] = new_frame
        # next states
        prev_stacked_frames = self.replay_buffer['next_states'][prev, :, :, 1:]
        new_frame = np.concatenate((prev_stacked_frames, state), axis=-1)
        self.replay_buffer['next_states'][start, :, :, :] = new_frame
        self.buffer_counter += 1

    def random_buffer_sample(self, states, actions, rewards, next_states, dones):
        """ Sample randomly a contiguous sequence of information"""
        # make sure that number of elements of SARS larger than
        # Buffer size
        if self.buffer_filled:
            # If buffer filled sample with replacement
            index = np.random.choice(self.buffer_size, self.batch_size, replace=True)
        elif self.buffer_counter < self.batch_size:
            # if we haven't filled up with self.batch_size, take all
            index = np.arange(self.buffer_counter)
            np.random.shuffle(index)
        else:
            # Else we sample from replacement from 0 to self.buffer_counter
            index = np.random.choice(self.buffer_counter, self.batch_size, replace=True)
        # Get random batch
        states = self.replay_buffer['states'][index]
        actions = self.replay_buffer['actions'][index]
        rewards = self.replay_buffer['rewards'][index]
        next_states = self.replay_buffer['next_states'][index]
        dones = self.replay_buffer['dones'][index]
        return states, actions, rewards, next_states, dones

    def train_network(self, states, actions, rewards, next_states, dones):
        """ do one training step and update parameters"""
        # We don't use the inputs, gets random sample from experience buffer
        # If there aren't enough in buffer, just pass the normal ones.
        self.counter += 1
        # Update target network
        if self.counter % self.copy_target == 0:
            self.update_target_network
        # If we have filled the buffer, set filled to true
        if self.buffer_counter >= self.buffer_size and not self.buffer_filled:
            print('Buffer filled')
            self.buffer_filled = True

        # Add things to buffer
        self.add_to_buffer(states, actions, rewards, next_states, dones)

        # Get random batch
        states, actions, rewards, next_states, dones = self.random_buffer_sample(states, actions, rewards, next_states, dones)
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones,
                   self.filter_1_target: self.params[0],
                   self.conv1_b_target: self.params[1],
                   self.filter_2_target: self.params[2],
                   self.conv2_b_target: self.params[3],
                   self.W1_target: self.params[4],
                   self.b1_target: self.params[5],
                   self.W_output_target: self.params[6],
                   self.b_output_target: self.params[7]}
        # Only save this every 1000 steps
        if self.counter % 1000 == 0:
            summaries, _ = self.sess.run([self.merged, self.train_step], feed_dict=feed_dict)
            self.summary_writer.add_summary(summaries, self.counter)
        else:
            self.sess.run(self.train_step, feed_dict=feed_dict)

    def save_estimator(self):
        """ Save the current estimator to save_path"""
        saver = tf.train.Saver()
        # Add placeholders and operations to collection
        # Placholders
        tf.add_to_collection('states', self.states)
        tf.add_to_collection('actions', self.actions)
        tf.add_to_collection('rewards', self.rewards)
        tf.add_to_collection('next_states', self.next_states)
        tf.add_to_collection('dones', self.dones)
        # All target parameters
        tf.add_to_collection('filter_1_target', self.filter_1_target)
        tf.add_to_collection('conv1_b_target', self.conv1_b_target)
        tf.add_to_collection('filter_2_target', self.filter_2_target)
        tf.add_to_collection('conv2_b_target', self.conv2_b_target)
        tf.add_to_collection('W1_target', self.W1_target)
        tf.add_to_collection('b1_target', self.b1_target)
        tf.add_to_collection('W_output_target', self.W_output_target)
        tf.add_to_collection('b_output_target', self.b_output_target)
        # Operations
        tf.add_to_collection('greedy', self.greedy)
        tf.add_to_collection('delta', self.delta)
        tf.add_to_collection('loss', self.loss)
        saver.save(self.sess, os.path.join(self.save_path, 'model.ckpt'))

def create_save_directory(name, learning_rate):
    """ Compose save path and create save directory if non-existent"""
    save_name = '{name}-lr{lr}'.format(name=name, lr=learning_rate)
    timestamp = str(int(time.time()))
    save_name = save_name + '_' + timestamp
    save_path = os.path.join(PATHS['save_dir'], save_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path

def plot_4tuple(state, next_state):
    """ plot a 4 tuple of the state"""
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 16 * 4))
    ax = ax.ravel()
    for i in range(0, 4):
        ax[i].imshow(np.squeeze(state)[:, :, i], cmap='gray')
        ax[i].axis('off')
    for i in range(0, 4):
        ax[i + 4].imshow(np.squeeze(next_state)[:, :, i], cmap='gray')
        ax[i + 4].axis('off')
    fig.savefig('test.png')

def eval_agent(sess, agent, env, process, episodes=1):
    """ Evaluate the agent in order to see how well it is doing"""
    # Hold information
    # keep track of the states and other things
    # So that we can evaluate the residual

    # Run algorithm
    for episode in range(episodes):
        temp_reward = 0
        temp_raw_reward = 0
        temp_loss = 0
        temp_residual = 0
        t = 0

        # reset environment
        state = process.process(sess, env.reset())

        # Initialise the input with 4 of the same frame
        temp_state_buffer = np.tile(state[np.newaxis, :, :], (1, 1, 1, 4))
        temp_next_state_buffer = np.tile(state[np.newaxis, :, :], (1, 1, 1, 4))

        while True:
            action = agent.greedy_action(temp_state_buffer)
            next_state, reward, done, info = env.step(action)
            # Change stacks
            next_state = process.process(sess, next_state)
            # next state
            temp_next_state_buffer = np.roll(temp_next_state_buffer, -1, axis=-1)
            temp_next_state_buffer[0, :, :, -1] = np.squeeze(next_state)
            # Update parameter
            temp_reward += reward * GAMMA**t
            temp_raw_reward += reward
            temp_loss, temp_residual = agent.get_loss(temp_state_buffer, action, reward, temp_next_state_buffer, done)
            state = next_state
            # state
            temp_state_buffer = np.roll(temp_state_buffer, -1, axis=-1)
            temp_state_buffer[0, :, :, -1] = np.squeeze(state)
            t += 1
            if done:
                return t, temp_reward, temp_raw_reward, temp_loss/t, temp_residual/t
