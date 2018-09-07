import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        self.actions = tf.placeholder(tf.int64, [None, 1])
        self.actions_one_hot = tf.one_hot(self.actions, depth=self.output_dim)
        self.Q = tf.reduce_sum(tf.multiply(self.actions_one_hot, self.Qout), -1)
        self.targetQ = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def inference(self):
        net = slim.fully_connected(self.state, 20)
        net = slim.fully_connected(net, self.output_dim)
        return net

    def learn(self, states, actions, rewards, next_states):
        next_q = self.sess.run(self.Qout, feed_dict={self.state: next_states})
        targetQ = rewards + np.reshape(self.gamma * np.max(next_q, axis=-1), [-1, 1])
        feed_dict = {self.state: states, self.actions: actions, self.targetQ: targetQ}
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def select_action(self, state):
        if np.random.rand() < self.greedy:
            return self.sess.run(self.predict, feed_dict={self.state: state})
        else:
            return np.random.randint(0, self.output_dim, size=len(state))


def extract_data():
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv('MSFT.csv', parse_dates=['Date'], date_parser=dateparse)
    df = df[['Date', 'Close']].sort_values(by='Date')
    df = df.rename(index=str, columns={"Date": "time", "Close": "price"})
    df = df.round({'price': 2})
    df.to_csv('microsoft.csv', index=False)


def load_data():
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv('microsoft.csv', parse_dates=['time'], date_parser=dateparse)
    return df['price'].tail(2000).values


def run_simulation(policy, cash=1000, num_stocks=50, prices=load_data(), hist=1000):
    share_value = prices[hist - 1]
    property = cash + num_stocks * share_value
    for i in range(hist, np.shape(prices)[0] - 1):
        current_state = np.asmatrix(np.hstack((prices[i - hist:i], cash, num_stocks)))
        [action] = policy.select_action(current_state)
        share_value = prices[i]
        if action == 0 and cash > share_value:
            cash -= share_value
            num_stocks += 1
        elif action == 1 and num_stocks > 0:
            cash += share_value
            num_stocks -= 1
        else:
            action = 2

        new_property = cash + share_value * num_stocks
        award = new_property - property
        new_state = np.asmatrix(np.hstack((prices[i + 1 - hist:i + 1], cash, num_stocks)))
        policy.learn(current_state, np.array([[action]]), np.array([[award]]), new_state)
    return new_property


def run_simulations(n):
    xs = []
    ys = []
    policy = QLearningDecisionPolicy(input_dim=1002, output_dim=3)
    for step in range(1, n+1):
        val = int(run_simulation(policy))
        print(step, val)
        xs.append(step)
        ys.append(val)

    plt.plot(xs, ys)
    plt.savefig("rl.jpg")
    plt.show()


if __name__ == '__main__':
    # run_simulations(5000)
    a = np.asmatrix()
