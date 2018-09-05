import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import pandas as pd

np.random.seed(0)


class DecisionPolicy():

    def select_action(self, state):
        pass


class RandomDecisionPolicy(DecisionPolicy):

    def __init__(self, actions):
        self.action = actions

    def select_action(self, state):
        return np.random.choice(self.action)


class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, input_dim, output_dim, greedy=0.95, gamma=0.3):
        self.greedy = greedy
        self.gamma = gamma
        self.output_dim = output_dim
        self.state = tf.placeholder(tf.float32, [None, input_dim])
        self.Qout = self.inference()
        self.predict = tf.argmax(self.Qout, -1)
        self.actions = tf.placeholder(tf.int64, [None, input_dim])
        self.actions_one_hot = tf.one_hot(self.actions, depth=self.output_dim)
        self.Q = tf.reduce_sum(tf.multiply(self.actions_one_hot, self.Qout), -1)
        self.targetQ = tf.placeholder(tf.float32)
        self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def inference(self):
        net = slim.fully_connected(self.state, 20)
        net = slim.fully_connected(net, self.output_dim)
        return net

    def learn(self, state, action, reward, next_state):
        next_q = self.sess.run(self.Qout, feed_dict={self.state: next_state})
        targetQ = reward + self.gamma * np.max(next_q, axis=-1)
        feed_dict = {self.state: state, self.actions: action, self.targetQ: targetQ}
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def select_action(self, state):
        if np.random.rand() < self.greedy:
            return self.sess.run(self.predict, feed_dict={self.state: state})
        else:
            return np.random.randint(0, self.output_dim, size=len(state))


dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('MSFT.csv', parse_dates=['Date'], date_parser=dateparse)
df = df[['Date', 'Close']].sort_values(by='Date')
df = df.rename(index=str, columns={"Date": "time", "Close": "price"})
df.to_csv('microsoft.csv', index_col = False)
