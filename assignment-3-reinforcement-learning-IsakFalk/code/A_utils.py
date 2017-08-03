"""
Helper utilities for problem A
"""

import time
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib as mlp
mlp.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style({'lines.linewidth': 2.0})
import gym
gym.envs.register(
    id='MyCartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=300,
)
import tensorflow as tf

from SETTINGS import GAMMA, MAX_EP_LEN, PATHS

import tf_utils

# Plotting

def plot_A(epochs, ep_len, reward, losses, title, save_path, residue=False):
    """ Plot the episode lengths, reward and losses"""
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(8.0, 12.0))
    plt.style.use('seaborn')
    # episode length
    axes[0].plot(epochs, ep_len, color='blue')
    axes[0].set_ylabel('episode length')
    axes[0].set_xlabel('epochs')
    axes[0].set_title('Mean episode length')
    # discounted total reward
    axes[1].plot(epochs, reward, color='red')
    axes[1].set_ylabel('discounted total reward')
    axes[1].set_xlabel('epochs')
    axes[1].set_title('Mean discounted total reward')
    # If residue, we plot that, otherwise losses (residue**2)
    axes[2].plot(epochs, losses, color='green')
    if residue:
        axes[2].set_ylabel('residue')
        axes[2].set_xlabel('epochs')
        axes[2].set_title('Bellman residue')
    else:
        axes[2].set_ylabel('loss')
        axes[2].set_xlabel('epochs')
        axes[2].set_title('L2 loss')

    fig.tight_layout()
    save_name = os.path.split(save_path)[-1]
    abs_path = os.path.join(save_path, save_name + '.png')
    fig.savefig(abs_path)
    print('Plot saved to {}'.format(abs_path))
    fig.clf()


def plot_A_2(episodes, ep_len, reward, residuals, losses, name, save_path):
    alpha = 0.02
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(15, 10))
    # Axes 0
    sns.tsplot(data=ep_len, time=episodes, value='episode length', ax=axes[0, 0], color='red', err_style="unit_traces", err_kws={'alpha': alpha})
    axes[0, 0].set_xlabel('episode')
    axes[0, 0].set_title('Episode length')
    # Axes 1
    sns.tsplot(data=reward, time=episodes, value='reward', ax=axes[0, 1], color='green', err_style="unit_traces", err_kws={'alpha': alpha})
    axes[0, 1].set_xlabel('episode')
    axes[0, 1].set_title('Total discounted reward')
    # Axes 2
    sns.tsplot(data=residuals, time=episodes, value='residue', ax=axes[1, 0], color='blue', err_style="unit_traces", err_kws={'alpha': alpha})
    axes[1, 0].set_xlabel('episode')
    axes[1, 0].set_title('Bellman residue')
    # Axes 3
    sns.tsplot(data=losses, time=episodes, value='loss', ax=axes[1, 1], color='magenta', err_style="unit_traces", err_kws={'alpha': alpha})
    axes[1, 1].set_xlabel('episode')
    axes[1, 1].set_title('Loss')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, name + '.png'))
    print('Plot saved to {}'.format(save_path + name + '.png'))
    fig.clf()

# Agents

class AgentBase(metaclass=ABCMeta):
    """
    Base class for the agent class

    Metaclass for the value function approximators.
    """

    @abstractmethod
    def eval_action(self, state):
        """ Output an action with respect to evaluation policy given the current state"""
        pass

    @abstractmethod
    def train_action(self, state):
        """ Output an action with respect to training policy given the current state"""
        pass

    @abstractmethod
    def update_policy(self, state, action, reward, next_state, done):
        """ Update policy given the online information.

        If doing replay or anything else, can fix it outside of this method.
        """
        pass


class LoadAgent(AgentBase):
    """ Load an agent and evaluate performance"""

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

    def eval_action(self, state):
        """ get the greedy action"""
        feed_dict = {self.states: np.reshape(state, [-1, 4])}
        return self.sess.run(self.greedy, feed_dict=feed_dict)[0]

    def train_action(self, state):
        pass

    def update_policy(self, state, action, reward, next_state, done):
        pass

    def get_statistics(self, t, state, action, reward, next_state, done):
        pass


class OnlineAgent(AgentBase):
    """ Online agent for learning policy"""

    def __init__(self, estimator, eps):
        self.estimator = estimator
        self.eps = eps

    def eval_action(self, state):
        """ Output an action with respect to evaluation policy given the current state"""
        action =  self.estimator.greedy_action(state)
        return action

    def train_action(self, state):
        """ Output an action with respect to training policy given the current state"""
        random = np.random.binomial(1, self.eps)
        if random:
            action = np.random.binomial(1, 0.5)
        else:
            action =  self.estimator.greedy_action(state)
        return action

    def update_policy(self, state, action, reward, next_state, done):
        """ Update policy given the online information."""
        state = np.reshape(state, [-1, 4])
        action = np.reshape(action, [-1, 1])
        reward = np.reshape(reward, [-1, 1])
        next_state = np.reshape(next_state, [-1, 4])
        done = np.reshape(done, [-1, 1])
        self.estimator.train_network(state, action, reward, next_state, done)

    def get_statistics(self, t, state, action, reward, next_state, done):
        """ Get statistics from the estimator"""
        state = np.reshape(state, [-1, 4])
        action = np.reshape(action, [-1, 1])
        reward = np.reshape(reward, [-1, 1])
        next_state = np.reshape(next_state, [-1, 4])
        done = np.reshape(done, [-1, 1])
        stats = self.estimator.get_statistics(t, state, action, reward, next_state, done)
        return stats

    def save_estimator(self):
        """ Save the estimator to file"""
        self.estimator.save_estimator()


class AgentA3(AgentBase):

    def __init__(self, estimator, batch_size, batch_dict):
        self.estimator = estimator
        # dictionary of the different information. Numpy arrays
        self.batch_dict = batch_dict
        self.batch_size = batch_size
        self.save_path = estimator.save_path
        self.size_data_set = self.batch_dict['states'].shape[0]

    def eval_action(self, state):
        """ Get action from the thing dependent on the state"""
        return self.estimator.greedy_action(state)

    def train_action(self, state):
        pass

    def update_policy(self, state, action, reward, next_state, done):
        """ Offline updates, so leave blank"""
        pass

    def train_estimator(self):
        """ Train the estimator for one whole epoch"""
        index = np.arange(self.size_data_set)
        np.random.shuffle(index)
        for i in range(int(self.size_data_set/self.batch_size)):
            # Get batch of SARS
            batch = index[i * self.batch_size: (i + 1) * self.batch_size]
            batch_states = self.batch_dict['states'][batch, :]
            batch_actions = self.batch_dict['actions'][batch, :]
            batch_rewards = self.batch_dict['rewards'][batch, :]
            batch_next_states = self.batch_dict['next_states'][batch, :]
            batch_dones = self.batch_dict['dones'][batch]
            # Train
            self.estimator.train_network(batch_states,
                                         batch_actions,
                                         batch_rewards,
                                         batch_next_states,
                                         batch_dones)

    def get_loss(self):
        """ Approximate the loss over 2000 samples"""
        index = np.random.permutation(np.arange(self.size_data_set))[0:2000]
        batch_states = self.batch_dict['states'][index, :]
        batch_actions = self.batch_dict['actions'][index, :]
        batch_rewards = self.batch_dict['rewards'][index, :]
        batch_next_states = self.batch_dict['next_states'][index, :]
        batch_dones = self.batch_dict['dones'][index]
        return self.estimator.get_loss(batch_states,
                                       batch_actions,
                                       batch_rewards,
                                       batch_next_states,
                                       batch_dones)

    def get_statistics(self, *args):
        return 0

    def write_summaries(self, epoch):
        """ Write summaries to file"""
        self.estimator.write_summaries(epoch, self.batch_dict)

    def save_estimator(self):
        """ Save the estimator to file"""
        self.estimator.save_estimator()

# Estimators

class Estimator(metaclass=ABCMeta):
    """
    Value Function approximator.

    Metaclass for the value function approximators.
    """

    @abstractmethod
    def train_network(self):
        pass

    @abstractmethod
    def save_estimator(self):
        pass


class Linear(Estimator):
    """ Linear estimator"""

    def __init__(self, sess, learning_rate, save_path, input_size=4, output_size=2):
        # Setting attributes
        self.sess = sess
        self.save_path = save_path

        # Placeholders for (states, actions, rewards, next_states) for updating
        # function approximator in Q-learning
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='states')
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
        self.next_states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='next_states')
        self.dones = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='dones')

        # Network
        W1 = tf.get_variable(name='W1',
                             shape=[input_size, output_size],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name='b1',
                             shape=[output_size],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.0))

        # Q-learning
        self.predictions = tf.matmul(self.states, W1) + b1
        self.predictions_target = tf.matmul(self.next_states, W1) + b1

        # Index actually occurred (state, action) pairs
        mask = tf.concat([1.0 - self.actions, self.actions], axis=1)
        Q_t = tf.reduce_sum(tf.multiply(self.predictions, mask), axis=1)

        # Maximum over actions for the next state Q-value function
        Q_max = tf.reduce_max(self.predictions_target, axis=1)

        # Target, we stop gradient flow
        terminate = 1 - tf.cast(self.dones, tf.float32)
        target = self.rewards + GAMMA * terminate * tf.stop_gradient(Q_max)

        # Squared loss
        self.loss = 0.5 * tf.reduce_mean(tf.square(target - Q_t), name='loss')
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Greedy policy
        self.greedy = tf.argmax(self.predictions, axis=1)

        # Summary writer
        W1_hist = tf.summary.histogram('W1', W1)
        b1_hist = tf.summary.histogram('b1', b1)
        loss_scalar = tf.summary.scalar('loss', self.loss)
        self.parameter_summary = tf.summary.merge([W1_hist, b1_hist, loss_scalar])
        self.parameter_summary_writer = tf.summary.FileWriter(os.path.join(self.save_path, 'summaries'))

        # Initializing the computational graph
        self.sess.run(tf.global_variables_initializer())

    def inference(self, state):
        """ Infer the best action from the state"""
        prediction = self.sess.run(self.predictions, feed_dict={self.states: state})
        return prediction

    def train_network(self, states, actions, rewards, next_states, dones):
        """ do one training step and update parameters"""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        self.sess.run(self.train_step, feed_dict=feed_dict)

    def greedy_action(self, state):
        """ Perform a greedy policy selection"""
        feed_dict={self.states: np.reshape(state, [-1, 4])}
        return self.sess.run(self.greedy, feed_dict=feed_dict)[0]

    def get_loss(self, states, actions, rewards, next_states, dones):
        """ Get the loss"""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def write_summaries(self, epoch, batch_dict):
        """ Add summaries"""
        index = np.random.permutation(np.arange(batch_dict['states'].shape[0]))[0:2000]
        batch_states = batch_dict['states'][index, :]
        batch_actions = batch_dict['actions'][index, :]
        batch_rewards = batch_dict['rewards'][index, :]
        batch_next_states = batch_dict['next_states'][index, :]
        batch_dones = batch_dict['dones'][index]
        feed_dict={self.states: batch_states,
                   self.actions: batch_actions,
                   self.rewards: batch_rewards,
                   self.next_states: batch_next_states,
                   self.dones: batch_dones}
        summaries = self.sess.run(self.parameter_summary, feed_dict=feed_dict)
        self.parameter_summary_writer.add_summary(summaries, epoch)

    def save_estimator(self):
        """ Save the current estimator to save_path"""
        saver = tf.train.Saver()
        # Add the relevant operations and tensors to collection
        tf.add_to_collection('states', self.states)
        tf.add_to_collection('greedy', self.greedy)
        saver.save(self.sess, os.path.join(self.save_path, 'model.ckpt'))


class OfflineNeuralNet(Estimator):
    """ A3 offline Neural Network"""

    def __init__(self,
                 sess,
                 learning_rate,
                 save_path,
                 input_size=4,
                 hidden_size=100,
                 output_size=2):
        # Setting attributes
        self.sess = sess
        self.save_path = save_path

        # Placeholders for (states, actions, rewards, next_states) for updating
        # function approximator in Q-learning
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='states')
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
        self.next_states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='next_states')
        self.dones = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='dones')

        # Network
        # First layer
        W1 = tf.get_variable(name='W1',
                             shape=[input_size, hidden_size],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name='b1',
                             shape=[hidden_size],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.0))

        h_1 = tf.nn.relu(tf.matmul(self.states, W1) + b1)
        h_1_target = tf.nn.relu(tf.matmul(self.next_states, W1) + b1)

        # Second layer
        W2 = tf.get_variable(name='W2',
                             shape=[hidden_size, output_size],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name='b2',
                             shape=[output_size],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.0))

        # Q-learning
        self.predictions = tf.matmul(h_1, W2) + b2
        self.predictions_target = tf.matmul(h_1_target, W2) + b2

        # Index actually occurred (state, action) pairs
        mask = tf.concat([1.0 - self.actions, self.actions], axis=1)
        Q_t = tf.reduce_sum(tf.multiply(self.predictions, mask), axis=1)

        # Maximum over actions for the next state Q-value function
        Q_max = tf.reduce_max(self.predictions_target, axis=1)

        # Target, we stop gradient flow
        terminate = 1 - tf.cast(self.dones, tf.float32)
        target = self.rewards + GAMMA * terminate * tf.stop_gradient(Q_max)

        # Squared loss
        self.loss = 0.5 * tf.reduce_mean(tf.square(target - Q_t), name='loss')
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Greedy policy
        self.greedy = tf.argmax(self.predictions, axis=1)

        # Summary writer
        W1_hist = tf.summary.histogram('W1', W1)
        b1_hist = tf.summary.histogram('b1', b1)
        W2_hist = tf.summary.histogram('W2', W2)
        b2_hist = tf.summary.histogram('b2', b2)
        loss_scalar = tf.summary.scalar('loss', self.loss)
        self.parameter_summary = tf.summary.merge([W1_hist, b1_hist, W2_hist, b2_hist, loss_scalar])
        self.parameter_summary_writer = tf.summary.FileWriter(os.path.join(self.save_path, 'summaries'))

        # Initializing the computational graph
        self.sess.run(tf.global_variables_initializer())

    def train_network(self, states, actions, rewards, next_states, dones):
        """ do one training step and update parameters"""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        self.sess.run(self.train_step, feed_dict=feed_dict)

    def greedy_action(self, state):
        """ Perform a greedy policy selection"""
        feed_dict={self.states: np.reshape(state, [-1, 4])}
        return self.sess.run(self.greedy, feed_dict=feed_dict)[0]

    def get_loss(self, states, actions, rewards, next_states, dones):
        """ Get the loss"""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def write_summaries(self, epoch, batch_dict):
        """ Add summaries"""
        index = np.random.permutation(np.arange(batch_dict['states'].shape[0]))[0:2000]
        batch_states = batch_dict['states'][index, :]
        batch_actions = batch_dict['actions'][index, :]
        batch_rewards = batch_dict['rewards'][index, :]
        batch_next_states = batch_dict['next_states'][index, :]
        batch_dones = batch_dict['dones'][index]
        feed_dict={self.states: batch_states,
                   self.actions: batch_actions,
                   self.rewards: batch_rewards,
                   self.next_states: batch_next_states,
                   self.dones: batch_dones}
        summaries = self.sess.run(self.parameter_summary, feed_dict=feed_dict)
        self.parameter_summary_writer.add_summary(summaries, epoch)

    def save_estimator(self):
        """ Save the current estimator to save_path"""
        saver = tf.train.Saver()
        # Add the relevant operations and tensors to collection
        tf.add_to_collection('states', self.states)
        tf.add_to_collection('greedy', self.greedy)
        saver.save(self.sess, os.path.join(self.save_path, 'model.ckpt'))


class OnlineNeuralNetVanilla(Estimator):
    """ Neural net estimator"""

    def __init__(self, sess, save_path, learning_rate, hidden_units, input_size=4, output_size=2):
        # Setting attributes
        self.sess = sess
        self.save_path = save_path

        # Placeholders for (states, actions, rewards, next_states) for updating
        # function approximator in Q-learning
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='states')
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
        self.next_states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='next_states')
        self.dones = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='dones')

        # Network
        # First layer
        W1 = tf.get_variable(name='W1',
                             shape=[input_size, hidden_units],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name='b1',
                             shape=[hidden_units],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.0))

        h_1 = tf.nn.relu(tf.matmul(self.states, W1) + b1)
        h_1_target = tf.nn.relu(tf.matmul(self.next_states, W1) + b1)

        # Second layer
        W2 = tf.get_variable(name='W2',
                             shape=[hidden_units, output_size],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name='b2',
                             shape=[output_size],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.0))

        # Q-learning
        self.predictions = tf.matmul(h_1, W2) + b2
        self.predictions_target = tf.matmul(h_1_target, W2) + b2

        # Index actually occurred (state, action) pairs
        mask = tf.concat([1.0 - self.actions, self.actions], axis=1)
        Q_t = tf.reduce_sum(tf.multiply(self.predictions, mask), axis=1)

        # Maximum over actions for the next state Q-value function
        Q_max = tf.reduce_max(self.predictions_target, axis=1)

        # Target, we stop gradient flow
        terminate = 1 - tf.cast(self.dones, tf.float32)
        target = self.rewards + GAMMA * terminate * tf.stop_gradient(Q_max)

        # Squared loss
        self.bellman_residue = target - Q_t
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.bellman_residue), name='loss')
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Greedy policy
        self.greedy = tf.argmax(self.predictions, axis=1)

        # Summary writer
        W1_hist = tf.summary.histogram('W1', W1)
        b1_hist = tf.summary.histogram('b1', b1)
        W2_hist = tf.summary.histogram('W2', W2)
        b2_hist = tf.summary.histogram('b2', b2)
        loss_scalar = tf.summary.scalar('loss', self.loss)
        self.parameter_summary = tf.summary.merge([W1_hist, b1_hist, W2_hist, b2_hist, loss_scalar])
        self.parameter_summary_writer = tf.summary.FileWriter(os.path.join(self.save_path, 'summaries'))

        # Initializing the computational graph
        self.sess.run(tf.global_variables_initializer())

    def train_network(self, states, actions, rewards, next_states, dones):
        """ do one training step and update parameters"""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        self.sess.run(self.train_step, feed_dict=feed_dict)

    def greedy_action(self, state):
        """ Perform a greedy policy selection"""
        feed_dict={self.states: np.reshape(state, [-1, 4])}
        return self.sess.run(self.greedy, feed_dict=feed_dict)[0]

    def get_statistics(self, t, states, actions, rewards, next_states, dones):
        """ Write statistics to file (summaries) and output statistics

        Output summary files to the write place and return the
        needed statistics, bellman residue and the loss."""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        summaries, residue = self.sess.run([self.parameter_summary, self.bellman_residue], feed_dict=feed_dict)
        self.parameter_summary_writer.add_summary(summaries, t)
        return residue

    def save_estimator(self):
        """ Save the current estimator to save_path"""
        saver = tf.train.Saver()
        # Add the relevant operations and tensors to collection
        tf.add_to_collection('states', self.states)
        tf.add_to_collection('greedy', self.greedy)
        saver.save(self.sess, os.path.join(self.save_path, 'model.ckpt'))


class OnlineNeuralNetExpReplay(Estimator):
    """ Q-network with Experience replay buffer"""

    def __init__(self, sess, save_path, learning_rate, hidden_units, input_size=4, output_size=2, buffer_size=1000, batch_size=8):
        # Setting attributes
        self.sess = sess
        self.save_path = save_path
        self.batch_size = batch_size

        # Experience Replay buffers
        self.replay_buffer = dict(states=np.zeros((buffer_size, 4), dtype=np.float32),
                                  actions=np.zeros((buffer_size, 1), dtype=np.float32),
                                  rewards=np.zeros((buffer_size, 1), dtype=np.float32),
                                  next_states=np.zeros((buffer_size, 4), dtype=np.float32),
                                  dones=np.zeros((buffer_size, 1), dtype=np.bool))
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.buffer_filled = False
        assert self.buffer_size >= self.batch_size, 'Need to have buffer_size >= batch_size'
        self.t = 0

        # Placeholders for (states, actions, rewards, next_states) for updating
        # function approximator in Q-learning
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='states')
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
        self.next_states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='next_states')
        self.dones = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='dones')

        # Decay learning rate
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = learning_rate
        end_learning_rate = 0.000001
        decay_steps = 100000
        learning_rate = tf.train.polynomial_decay(starter_learning_rate,
                                                  global_step,
                                                  decay_steps,
                                                  end_learning_rate,
                                                  power=0.5)

        # Network
        # First layer
        W1 = tf.get_variable(name='W1',
                             shape=[input_size, hidden_units],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name='b1',
                             shape=[hidden_units],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.1))

        h_1 = tf.nn.relu(tf.matmul(self.states, W1) + b1)
        h_1_target = tf.nn.relu(tf.matmul(self.next_states, W1) + b1)

        # Second layer
        W2 = tf.get_variable(name='W2',
                             shape=[hidden_units, output_size],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name='b2',
                             shape=[output_size],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.1))

        # Q-learning
        self.predictions = tf.matmul(h_1, W2) + b2
        self.predictions_target = tf.matmul(h_1_target, W2) + b2

        # Index actually occurred (state, action) pairs
        mask = tf.concat([1.0 - self.actions, self.actions], axis=1)
        Q_t = tf.reduce_sum(tf.multiply(self.predictions, mask), axis=1)

        # Maximum over actions for the next state Q-value function
        Q_max = tf.reduce_max(self.predictions_target, axis=1)

        # Target, we stop gradient flow
        terminate = 1 - tf.cast(self.dones, tf.float32)
        target = self.rewards + GAMMA * terminate * tf.stop_gradient(Q_max)

        # Squared loss
        self.bellman_residue = target - Q_t
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.bellman_residue), name='loss')
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

        # Greedy policy
        self.greedy = tf.argmax(self.predictions, axis=1)

        # Make summary directory
        os.makedirs(os.path.join(self.save_path, 'summaries'), exist_ok=True)

        # Summary writer
        W1_hist = tf.summary.histogram('W1', W1)
        b1_hist = tf.summary.histogram('b1', b1)
        W2_hist = tf.summary.histogram('W2', W2)
        b2_hist = tf.summary.histogram('b2', b2)
        loss_scalar = tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge([W1_hist, b1_hist, W2_hist, b2_hist, loss_scalar])
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.save_path, 'summaries'))

        # Initializing the computational graph
        self.sess.run(tf.global_variables_initializer())

    def add_to_buffer(self, states, actions, rewards, next_states, dones):
        """ Add the lists from the environment of the episode to the replay buffer"""
        # number of steps taken in previous episode
        ep_len = len(actions)
        start = self.buffer_counter
        end = start + ep_len
        # index lists
        index = [i % self.buffer_size for i in range(start, end)]
        # add to buffer
        self.replay_buffer['states'][index, :] = np.array(states).reshape([-1, 4])
        self.replay_buffer['actions'][index, :] = np.reshape(actions, [-1, 1])
        self.replay_buffer['rewards'][index, :] = np.reshape(rewards, [-1, 1])
        self.replay_buffer['next_states'][index, :] = np.array(next_states).reshape([-1, 4])
        self.replay_buffer['dones'][index, :] = np.reshape(dones, [-1, 1])
        # increment buffer counter
        self.buffer_counter += ep_len
        if self.buffer_counter > self.buffer_size and not self.buffer_filled:
            # Filled up buffer
            self.buffer_filled = True
            print('Buffer filled')
        # Buffer counter always modulo 'self.buffer_size'
        self.buffer_counter %= self.buffer_size

    def random_buffer_sample(self, states, actions, rewards, next_states, dones):
        """ Sample randomly a contiguous sequence of information"""
        # make sure that number of elements of SARS larger than
        # Buffer size
        if self.buffer_filled:
            # If buffer filled sample with replacement
            index = np.random.choice(self.buffer_size, self.batch_size, replace=True)
        elif self.buffer_counter == 0:
            # If we are in first episode just do online
            return states.reshape([-1, 4]), actions, rewards, next_states.reshape([-1, 4]), dones
        elif self.buffer_counter < self.batch_size:
            # if we haven't filled up with self.batch_size, take all
            index = np.arange(self.buffer_counter)
            np.random.shuffle(index)
        else:
            # Else we sample from replacement from 0 to self.buffer_counter
            index = np.random.choice(self.buffer_counter, self.batch_size, replace=True)
        # Get random batch
        states = self.replay_buffer['states'][index, :]
        actions = self.replay_buffer['actions'][index, :]
        rewards = self.replay_buffer['rewards'][index, :]
        next_states = self.replay_buffer['next_states'][index, :]
        dones = self.replay_buffer['dones'][index, :]
        return states, actions, rewards, next_states, dones

    def train_network(self, states, actions, rewards, next_states, dones):
        """ do one training step and update parameters"""
        # Increment counter
        self.t += 1
        # We don't use the inputs, gets random sample from experience buffer
        # If there aren't enough in buffer, just pass the normal ones.
        states, actions, rewards, next_states, dones = self.random_buffer_sample(states, actions, rewards, next_states, dones)
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        summaries, _ = self.sess.run([self.merged, self.train_step], feed_dict=feed_dict)
        self.summary_writer.add_summary(summaries, self.t)

    def greedy_action(self, state):
        """ Perform a greedy policy selection"""
        feed_dict = {self.states: np.reshape(state, [-1, 4])}
        action = self.sess.run(self.greedy, feed_dict=feed_dict)[0]
        return action

    def get_statistics(self, t, states, actions, rewards, next_states, dones):
        """ Write statistics to file (summaries) and output statistics

        Output summary files to the write place and return the
        needed statistics, bellman residue and the loss."""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        residue = self.sess.run(self.bellman_residue, feed_dict=feed_dict)
        return residue

    def save_estimator(self):
        """ Save the current estimator to save_path"""
        saver = tf.train.Saver()
        # Add the relevant operations and tensors to collection
        tf.add_to_collection('states', self.states)
        tf.add_to_collection('greedy', self.greedy)
        saver.save(self.sess, os.path.join(self.save_path, 'model.ckpt'))


class OnlineNeuralNetTargetReplay(Estimator):
    """ Q-network with Experience replay buffer and target network."""

    def __init__(self, sess, save_path, learning_rate, hidden_units, input_size=4, output_size=2, buffer_size=1000, batch_size=8, copy_target=5):
        # Setting attributes
        self.sess = sess
        self.save_path = save_path
        self.batch_size = batch_size
        self.copy_target = copy_target
        # Holding the parameters
        self.params = []
        # Counter for number of episodes ran
        self.counter = 0
        # number of iterations
        self.t = 0

        # Experience Replay buffers
        self.replay_buffer = dict(states=np.zeros((buffer_size, 4), dtype=np.float32),
                                  actions=np.zeros((buffer_size, 1), dtype=np.float32),
                                  rewards=np.zeros((buffer_size, 1), dtype=np.float32),
                                  next_states=np.zeros((buffer_size, 4), dtype=np.float32),
                                  dones=np.zeros((buffer_size, 1), dtype=np.bool))
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.buffer_filled = False
        assert self.buffer_size >= self.batch_size, 'Need to have buffer_size >= batch_size'

        # Placeholders for (states, actions, rewards, next_states) for updating
        # function approximator in Q-learning
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='states')
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
        self.next_states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='next_states')
        self.dones = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='dones')

        # Target network placeholders
        self.W1_target = tf.placeholder(dtype=tf.float32, shape=[input_size, hidden_units], name='W1_target')
        self.b1_target = tf.placeholder(dtype=tf.float32, shape=[hidden_units], name='b1_target')
        self.W2_target = tf.placeholder(dtype=tf.float32, shape=[hidden_units, output_size], name='W2_target')
        self.b2_target = tf.placeholder(dtype=tf.float32, shape=[output_size], name='b2_target')

        # # Decay learning
        global_step = tf.Variable(0, trainable=False)
        # starter_learning_rate = learning_rate
        # end_learning_rate = 0.000001
        # decay_steps = 100000
        # learning_rate = tf.train.polynomial_decay(starter_learning_rate,
        #                                           global_step,
        #                                           decay_steps,
        #                                           end_learning_rate,
        #                                           power=0.5)

        # Network
        # First layer
        self.W1 = tf.get_variable(name='W1',
                                  shape=[input_size, hidden_units],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.b1 = tf.get_variable(name='b1',
                                  shape=[hidden_units],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(value=0.1))

        # Output
        h_1 = tf.nn.relu(tf.matmul(self.states, self.W1) + self.b1)

        # Target
        h_1_target = tf.nn.relu(tf.matmul(self.next_states, self.W1_target) + self.b1_target)

        # Second layer
        self.W2 = tf.get_variable(name='W2',
                             shape=[hidden_units, output_size],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        self.b2 = tf.get_variable(name='b2',
                             shape=[output_size],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.0))

        # Q-learning
        self.predictions = tf.matmul(h_1, self.W2) + self.b2
        self.predictions_target = tf.matmul(h_1_target, self.W2_target) + self.b2_target

        # Index actually occurred (state, action) pairs
        mask = tf.concat([1.0 - self.actions, self.actions], axis=1)
        Q_t = tf.reduce_sum(tf.multiply(self.predictions, mask), axis=1)

        # Maximum over actions for the next state Q-value function
        Q_max = tf.reduce_max(self.predictions_target, axis=1)

        # Target, we stop gradient flow
        terminate = 1 - tf.cast(self.dones, tf.float32)
        target = self.rewards + GAMMA * terminate * tf.stop_gradient(Q_max)

        # Squared loss
        self.bellman_residue = target - Q_t
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.bellman_residue), name='loss')
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

        # Greedy policy
        self.greedy = tf.argmax(self.predictions, axis=1)

        # Make summary directory
        os.makedirs(os.path.join(self.save_path, 'summaries'), exist_ok=True)

        # Summary writer
        W1_hist = tf.summary.histogram('W1', self.W1)
        b1_hist = tf.summary.histogram('b1', self.b1)
        W2_hist = tf.summary.histogram('W2', self.W2)
        b2_hist = tf.summary.histogram('b2', self.b2)
        loss_scalar = tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge([W1_hist, b1_hist, W2_hist, b2_hist, loss_scalar])
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.save_path, 'summaries'))

        # Initializing the computational graph
        self.sess.run(tf.global_variables_initializer())

        # Get params
        self.params = self.sess.run([self.W1, self.b1, self.W2, self.b2])

    def update_target_network(self):
        """ Update the target network"""
        # increment counter
        self.counter += 1
        # Copy the network parameters
        if self.counter % self.copy_target == 0:
            self.params = self.sess.run([self.W1, self.b1, self.W2, self.b2])

    def random_buffer_sample(self, states, actions, rewards, next_states, dones):
        """ Sample randomly a contiguous sequence of information"""
        # make sure that number of elements of SARS larger than
        # Buffer size
        if self.buffer_filled:
            # If buffer filled sample with replacement
            index = np.random.choice(self.buffer_size, self.batch_size, replace=True)
        elif self.buffer_counter == 0:
            # If we are in first episode just do online
            return states.reshape([-1, 4]), actions, rewards, next_states.reshape([-1, 4]), dones
        elif self.buffer_counter < self.batch_size:
            # if we haven't filled up with self.batch_size, take all
            index = np.arange(self.buffer_counter)
            np.random.shuffle(index)
        else:
            # Else we sample from replacement from 0 to self.buffer_counter
            index = np.random.choice(self.buffer_counter, self.batch_size, replace=True)
        # Get random batch
        states = self.replay_buffer['states'][index, :]
        actions = self.replay_buffer['actions'][index, :]
        rewards = self.replay_buffer['rewards'][index, :]
        next_states = self.replay_buffer['next_states'][index, :]
        dones = self.replay_buffer['dones'][index, :]
        return states, actions, rewards, next_states, dones

    def add_to_buffer(self, states, actions, rewards, next_states, dones):
        """ Add the lists from the environment of the episode to the replay buffer"""
        # number of steps taken in previous episode
        ep_len = len(actions)
        start = self.buffer_counter
        end = start + ep_len
        # index lists
        index = [i % self.buffer_size for i in range(start, end)]
        # add to buffer
        self.replay_buffer['states'][index, :] = np.array(states).reshape([-1, 4])
        self.replay_buffer['actions'][index, :] = np.reshape(actions, [-1, 1])
        self.replay_buffer['rewards'][index, :] = np.reshape(rewards, [-1, 1])
        self.replay_buffer['next_states'][index, :] = np.array(next_states).reshape([-1, 4])
        self.replay_buffer['dones'][index, :] = np.reshape(dones, [-1, 1])
        # increment buffer counter
        self.buffer_counter += ep_len
        if self.buffer_counter > self.buffer_size and not self.buffer_filled:
            # Filled up buffer
            print('Buffer filled')
            self.buffer_filled = True
        # Buffer counter always modulo 'self.buffer_size'
        self.buffer_counter %= self.buffer_size

    def train_network(self, states, actions, rewards, next_states, dones):
        """ do one training step and update parameters"""
        # Increment number of iterations run
        self.t += 1
        # We don't use the inputs, gets random sample from experience buffer
        # If there aren't enough in buffer, just pass the normal ones.
        states, actions, rewards, next_states, dones = self.random_buffer_sample(states, actions, rewards, next_states, dones)
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones,
                   self.W1_target: self.params[0],
                   self.b1_target: self.params[1],
                   self.W2_target: self.params[2],
                   self.b2_target: self.params[3]}
        summaries, _ = self.sess.run([self.merged, self.train_step], feed_dict=feed_dict)
        self.summary_writer.add_summary(summaries, self.t)

    def greedy_action(self, state):
        """ Perform a greedy policy selection"""
        feed_dict={self.states: np.reshape(state, [-1, 4])}
        return self.sess.run(self.greedy, feed_dict=feed_dict)[0]

    def get_statistics(self, t, states, actions, rewards, next_states, dones):
        """ Write statistics to file (summaries) and output statistics

        Output summary files to the write place and return the
        needed statistics, bellman residue and the loss."""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones,
                   self.W1_target: self.params[0],
                   self.b1_target: self.params[1],
                   self.W2_target: self.params[2],
                   self.b2_target: self.params[3]}
        residue = self.sess.run(self.bellman_residue, feed_dict=feed_dict)
        return residue

    def save_estimator(self):
        """ Save the current estimator to save_path"""
        saver = tf.train.Saver()
        # Add the relevant operations and tensors to collection
        tf.add_to_collection('states', self.states)
        tf.add_to_collection('greedy', self.greedy)
        saver.save(self.sess, os.path.join(self.save_path, 'model.ckpt'))


class SARSA(Estimator):
    """ Neural net estimator"""

    def __init__(self, sess, save_path, learning_rate, hidden_units, input_size=4, output_size=2):
        # Setting attributes
        self.sess = sess
        self.save_path = save_path

        # Placeholders for (states, actions, rewards, next_states) for updating
        # function approximator in Q-learning
        self.states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='states')
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='rewards')
        self.next_states = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='next_states')
        self.dones = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='dones')

        # Network
        # First layer
        W1 = tf.get_variable(name='W1',
                             shape=[input_size, hidden_units],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name='b1',
                             shape=[hidden_units],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.0))

        h_1 = tf.nn.relu(tf.matmul(self.states, W1) + b1)
        h_1_target = tf.nn.relu(tf.matmul(self.next_states, W1) + b1)

        # Second layer
        W2 = tf.get_variable(name='W2',
                             shape=[hidden_units, output_size],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name='b2',
                             shape=[output_size],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(value=0.0))

        # Q-learning
        self.predictions = tf.matmul(h_1, W2) + b2
        self.predictions_target = tf.matmul(h_1_target, W2) + b2

        # Index actually occurred (state, action) pairs
        mask = tf.concat([1.0 - self.actions, self.actions], axis=1)
        Q_t = tf.reduce_sum(tf.multiply(self.predictions, mask), axis=1)

        # Maximum over actions for the next state Q-value function
        Q_max = tf.reduce_max(self.predictions_target, axis=1)

        # Target, we stop gradient flow
        terminate = 1 - tf.cast(self.dones, tf.float32)
        target = self.rewards + GAMMA * terminate * tf.stop_gradient(Q_max)

        # Squared loss
        self.bellman_residue = target - Q_t
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.bellman_residue), name='loss')
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Greedy policy
        self.greedy = tf.argmax(self.predictions, axis=1)

        # Summary writer
        W1_hist = tf.summary.histogram('W1', W1)
        b1_hist = tf.summary.histogram('b1', b1)
        W2_hist = tf.summary.histogram('W2', W2)
        b2_hist = tf.summary.histogram('b2', b2)
        loss_scalar = tf.summary.scalar('loss', self.loss)
        self.parameter_summary = tf.summary.merge([W1_hist, b1_hist, W2_hist, b2_hist, loss_scalar])
        self.parameter_summary_writer = tf.summary.FileWriter(os.path.join(self.save_path, 'summaries'))

        # Initializing the computational graph
        self.sess.run(tf.global_variables_initializer())

    def train_network(self, states, actions, rewards, next_states, dones):
        """ do one training step and update parameters"""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        self.sess.run(self.train_step, feed_dict=feed_dict)

    def greedy_action(self, state):
        """ Perform a greedy policy selection"""
        feed_dict={self.states: np.reshape(state, [-1, 4])}
        return self.sess.run(self.greedy, feed_dict=feed_dict)[0]

    def get_statistics(self, t, states, actions, rewards, next_states, dones):
        """ Write statistics to file (summaries) and output statistics

        Output summary files to the write place and return the
        needed statistics, bellman residue and the loss."""
        feed_dict={self.states: states,
                   self.actions: actions,
                   self.rewards: rewards,
                   self.next_states: next_states,
                   self.dones: dones}
        summaries, residue = self.sess.run([self.parameter_summary, self.bellman_residue], feed_dict=feed_dict)
        self.parameter_summary_writer.add_summary(summaries, t)
        return residue

    def save_estimator(self):
        """ Save the current estimator to save_path"""
        saver = tf.train.Saver()
        # Add the relevant operations and tensors to collection
        tf.add_to_collection('states', self.states)
        tf.add_to_collection('greedy', self.greedy)
        saver.save(self.sess, os.path.join(self.save_path, 'model.ckpt'))


class CartPoleEnv():
    """ Cart pole environment class"""

    def __init__(self, max_ep_len=MAX_EP_LEN, gamma=GAMMA):
        """ Instantiate the environment

        We keep lists around (ep_len, disc_total_rew) that
        holds the information on episode length and discounted
        total reward for each run over a number of episodes.

        Args:
            max_ep_len: (int) maximum number of episodes
            gamma: (float) discount factor in [0, 1]
            render: (bool) render video
            print_terminal: (bool) print to terminal (stdout)"""
        self.max_ep_len = max_ep_len
        self.gamma = gamma
        self.env = gym.make('MyCartPole-v0')

        # Hold the ep_len and reward for each episode
        self.ep_len = []
        self.tot_rew = []

        # Keep track of the needed information
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def step(self, action):
        """ Step one timestep forward

        Change the behaviour of the reward
        function as specified in the assignment
        sheet.
        """
        observation, _, done, info = self.env.step(action)
        return observation, -int(done), done, info

    def get_holders(self):
        """ Return the holder lists"""
        return self.states, self.actions, self.rewards, self.next_states, self.dones

    def reset_holders(self):
        """ Reset the holders of information"""
        # MDP
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def reset(self):
        """ Reset the environment"""
        self.reset_holders()
        return self.env.reset()

    def run_update_episode(self, agent):
        """ Run an episode using the agent.

        We copy the function from the docs of the open AI.
        This requires a consistent interface from the agent.
        Essentially the agent have to have the get_action and
        update_policy in order to work."""
        state = self.reset()
        for t in range(MAX_EP_LEN):
            action = agent.train_action(state)
            next_state, reward, done, info = self.step(action)
            self.states.append(state.tolist())
            self.next_states.append(next_state.tolist())
            self.rewards.append(reward)
            self.actions.append(action)
            self.dones.append(done)
            agent.update_policy(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    def evaluate_performance(self, agent, num_episodes=100, count=0):
        """ Evaluate performance of the current agent.

        Runs num_episodes in order to get the average reward and
        episode lengths using the action policy of the supplied agent.

        Args:
            agent: agent with a policy
            num_episodes: over the number of episodes to evaluate over

        Returns:
            mean_ep_len: average episode length
            mean_reward: average total discounted reward
        """
        # Hold information
        list_ep_len = []
        list_reward = []

        # keep track of the states and other things
        # So that we can evaluate the residual
        eval_states = []
        eval_actions = []
        eval_rewards = []
        eval_next_states = []
        eval_dones = []

        # Average over num_episodes
        for _ in range(num_episodes):
            tot_disc_reward = 0
            state = self.env.reset()
            for t in range(MAX_EP_LEN):
                action = agent.eval_action(state)
                next_state, reward, done, info = self.step(action)
                tot_disc_reward += reward * GAMMA**t
                eval_states.append(state)
                eval_actions.append(action)
                eval_rewards.append(reward)
                eval_next_states.append(next_state)
                eval_dones.append(done)
                state = next_state
                if done:
                    list_ep_len.append(t+1)
                    list_reward.append(tot_disc_reward)
                    break

        # Average these things
        mean_ep_len = np.mean(list_ep_len)
        mean_reward = np.mean(list_reward)
        residual = agent.get_statistics(count,
                                        eval_states,
                                        eval_actions,
                                        eval_rewards,
                                        eval_next_states,
                                        eval_dones)

        return mean_ep_len, mean_reward, residual


def create_random_batch(max_steps=MAX_EP_LEN, num_episodes=2000):
    """ Generate a batch of (s_t, a_t, r_t+1, s_t+1) tuples unsing a uniform policy.

    Returns:
        Histoy: list of elements of the tuples (s_t, a_t, r_t+1, s_t+1) where
                s_t (a, b, c, d), a_t (0, 1), r_t+1 (0, -1) and
                s_t+1 (a, b, c, d)."""
    # Cartpole environment
    env = CartPoleEnv()

    # Hold the history
    states = []
    rewards = []
    actions = []
    dones = []
    next_states = []

    for i_episode in range(num_episodes):
        state = env.reset()
        for t in range(max_steps):
            action = env.env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            state = next_state
            if done:
                break
    print('Finished sampling {} data points.'.format(len(rewards)))
    # Make arrays
    states = np.asarray(states)
    actions = np.reshape(np.asarray(actions), [-1, 1])
    rewards = np.reshape(np.asarray(rewards), [-1, 1])
    next_states = np.asarray(next_states)
    dones = np.reshape(np.asarray(dones), [-1, 1])
    return states, actions, rewards, next_states, dones


def create_save_directory(name, learning_rate):
    """ Compose save path and create save directory if non-existent"""
    save_name = '{name}-lr{lr}'.format(name=name, lr=learning_rate)
    timestamp = str(int(time.time()))
    save_name = save_name + '_' + timestamp
    save_path = os.path.join(PATHS['save_dir'], save_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path
