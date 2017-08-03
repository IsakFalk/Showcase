import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import gym

env = gym.make('CartPole-v0')
max_ep_length = 300
nr_ep = 100
episode_length = np.zeros(nr_ep)
returns = np.zeros(nr_ep)
discount = 0.99

def masking(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)

class Estimator():
    """
    Value Function approximator.
    """

class NeuralNetwork(Estimator):

    def __init__(self, input_size, hidden_size, learning_rate):
        print('Initializing NeuralNetwork Estimator with input size: %d, hidden units: %d and learning rate: %d' % (input_size, hidden_size, learning_rate))
        self.x = tf.placeholder(tf.float32, shape=(None, input_size), name='input_ph')
        self.target = tf.placeholder(tf.float32, shape=(None, 1), name='target_ph')
        self.mask = tf.placeholder(tf.float32, shape=(None, 2), name='mask_ph')

        W1 = tf.get_variable("W1", [input_size, hidden_size])
        b1 = tf.get_variable(dtype=tf.float32, shape=[hidden_size], initializer=tf.constant_initializer(value=0.1), name="b1")
        hidden = tf.nn.relu(tf.matmul(self.x, W1) + b1)

        W2 = tf.get_variable("W2", [hidden_size, 2])
        b2 = tf.get_variable(dtype=tf.float32, shape=[2], initializer=tf.constant_initializer(value=0.1), name="b2")
        self.prediction = tf.matmul(hidden, W2) + b2

        out_train = tf.reduce_sum(tf.multiply(self.prediction, self.mask), axis=1)
        self.loss = tf.nn.l2_loss(self.target - out_train)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.weights = [W1, b1, W2, b2]
        self.target_weights = self.sess.run(self.weights)

    def inference(self, x, use_target):
        with self.sess.as_default():
            if use_target:
                feed_dict = {self.x: x}
                feed_dict.update(zip(self.weights, self.target_weights))
                inferred = self.sess.run(self.prediction, feed_dict=feed_dict)
            else:
                inferred = self.sess.run(self.prediction, feed_dict={self.x: x})
        return inferred

    def inference_target(self, x):
        with self.sess.as_default():
            feed_dict = {self.x: x}
            feed_dict.update(zip(self.weights, self.target_weights))
            inferred = self.sess.run(self.prediction, feed_dict = feed_dict)

    def learn(self, X, y, mask):
        with self.sess.as_default():
            self.sess.run(self.train_step, feed_dict={self.x: X, self.mask: mask, self.target: y})

    def update(self, state, action, td_target):
        mask = masking(action).squeeze(axis=1)
        self.learn(state, td_target, mask)

    def predict(self, state, for_target=False):
        return self.inference(state, for_target)

    def eval(self, state, action, td_target):
        mask = masking(action).squeeze(axis=1)
        return self.sess.run(self.loss, feed_dict={self.x: state, self.mask: mask, self.target: td_target})

    def update_target(self):
        self.target_weights = self.sess.run(self.weights)

def create_random_batch(env, estimator, n_episodes, discount=0.99, max_steps=300):
    """Chooses an action based on a random uniform policy, saves the state, action, reward, and next_state
    Returns:
        History: list with elements [state (a,b,c,d), action (0,1), reward (0,-1), next_state(a,b,c,d)]
    """
    print('Getting %d episodes under uniform random policy...' % n_episodes)
    states = []
    rewards = []
    actions = []
    dones = []
    next_states = []

    for i_episode in range(n_episodes):
        state = env.reset()
        for t in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            reward = 0
            if done:
                reward = -1
            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            state = next_state
            if done:
                break
    print('Finished sampling.')
    return states, actions, rewards, next_states, dones

def e_greedy(observation, estimator, epsilon, nr_actions, greedy):
    A = np.ones(nr_actions, dtype=float) * epsilon / nr_actions
    q_values = estimator.predict(observation)
    best_action = np.argmax(q_values)
    if greedy:
        A[best_action] += (1.0 - epsilon)
        action = np.random.choice(np.arange(len(A)), p=A)
    else:
        action = best_action
    return int(action)

def run_agent(env, estimator, nr_ep):
    episode_length = np.zeros(nr_ep)
    returns = np.zeros(nr_ep)
    print('Running an agent...')
    for i_episode in range(nr_ep):
        state = env.reset()
        for t in range(300):
            action = e_greedy(np.asarray([state]), estimator = estimator, epsilon = 0.0001, nr_actions = 2, greedy = True)
            next_state, reward, done, _ = env.step(action)
            if done or t == 300:
                episode_length[i_episode] = t+1
                returns[i_episode] = reward*np.power(discount, t)
                break
            state = next_state
    return np.mean(episode_length), np.mean(returns)

def plot(losses, episode_lengths, reward_list):
    plt.title('Loss')
    plt.subplot(3,1,1)
    plt.plot(losses)
    plt.title('Mean Episode Lengths')
    plt.subplot(3,1,2)
    plt.plot(episode_lengths)
    plt.subplot(3,1,3)
    plt.title('Mean rewards')
    plt.plot(reward_list)
    plt.show()

def batch_offline_QLearning(env, estimator, epochs, batch_size, discount):
    states, actions, rewards, next_states, dones = create_random_batch(env, estimator, n_episodes = 2000, discount=0.99, max_steps=300)

    n_samples = len(states)

    losses = []
    episode_lengths= []
    reward_list = []
    print('Starting training...')
    for epoch in range(epochs):
        for i in range(n_samples // batch_size):
            #Transposing and adding an extra dimension.
            actions_batch = np.transpose(np.asarray([actions[i*batch_size:(i+1)*batch_size]]))
            reward_batch = np.transpose(np.asarray([rewards[i*batch_size:(i+1)*batch_size]]))
            done_batch = np.transpose(np.asarray([dones[i*batch_size:(i+1)*batch_size]]))

            next_states_batch = np.asarray(next_states[i*batch_size:(i+1)*batch_size])
            states_batch = np.asarray(states[i*batch_size:(i+1)*batch_size])

            q_values_next = estimator.predict(next_states_batch, for_target=True)
            target_actions = np.argmax(q_values_next, axis=1)[:, None]
            td_target = reward_batch + discount * np.max(q_values_next, axis=1, keepdims=True) * (1 - done_batch.astype(np.int))

            estimator.update(states_batch, actions_batch, td_target)

            if i == 0 and epoch % 10 == 0:
                loss = estimator.eval(states_batch, actions_batch, td_target)
                episode_length, reward_agent = run_agent(env, estimator, 10)

                losses.append(loss)
                episode_lengths.append(episode_length)
                reward_list.append(reward_agent)
                print("Epoch: %d | Loss: %.4f | Rewards: %.4f | Episode lengths: %d" % (epoch, loss, reward_agent, episode_length))

    mean_episode, mean_reward = run_agent(env, estimator, 10)

    plot(losses, episode_lengths, reward_list)
    print(mean_episode, mean_reward, 'mean length/reward')
    return mean_episode, mean_reward, losses, episode_lengths, reward_list

estimator = NeuralNetwork(4, 100, 0.00001)
mean_episode, mean_reward, losses, episode_lengths, reward_list = batch_offline_QLearning(env, estimator, 500, 100, 0.99)



