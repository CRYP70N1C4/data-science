import numpy as np
import random, game_2048
from collections import deque
import tensorflow as tf
from tensorflow.contrib import slim
import time


class QNetwork():
    def __init__(self, input_size=16, output_size=4):
        self.input_size = input_size
        self.output_size = output_size
        self.env_input = tf.placeholder(tf.float32, [None, input_size])
        self.Qout = self.__cnn()
        self.predict = tf.argmax(self.Qout, -1)
        self.targetQ = tf.placeholder(dtype=tf.float32)
        self.actions = tf.placeholder(dtype=tf.int32)
        self.Q = tf.reduce_sum(tf.one_hot(self.actions, depth=output_size, dtype=tf.float32) * self.Qout,
                               reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

    def __mlp(self):
        net = slim.fully_connected(self.env_input, 512)
        net = slim.fully_connected(net, 128)
        net = slim.fully_connected(net, self.output_size, activation_fn=None)
        return net

    def __cnn(self):
        net = tf.reshape(self.env_input, [-1, 4, 4, 1])
        net = slim.conv2d(net, 32, [2, 2])
        net = slim.conv2d(net, 32, [2, 2])
        net = slim.flatten(net)
        net = slim.fully_connected(net, 128, activation_fn=None)
        net = slim.fully_connected(net, 4, activation_fn=None)
        return net

    def get_predict(self, session, env_input):
        return session.run(self.predict, feed_dict={self.env_input: env_input})

    def get_Qout(self, session, env_input):
        return session.run(self.Qout, feed_dict={self.env_input: env_input})

    def learn(self, session, env_input, actions, targetQ):
        session.run(self.train_op, feed_dict={self.env_input: env_input, self.actions: actions, self.targetQ: targetQ})


class replay_buffer():
    def __init__(self, buffer_size=20000):
        self.buffer = deque()
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def add_all(self, buffer: deque):
        self.buffer.extend(buffer)
        while len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def sample(self, size):
        return np.array(random.sample(self.buffer, size))


def update_target_graph(tf_vars, tau):
    mid = len(tf_vars) // 2
    op_holders = []
    for i in range(mid):
        val = (1 - tau) * tf_vars[i + mid].value() + tau * tf_vars[i].value()
        op_holders.append(tf_vars[i + mid].assign(val))
    return op_holders


env = game_2048.Game()
gloal_buffer = replay_buffer()
np.random.seed(0)
greedy = 0.9
eposide_num = 100000000
max_step = 200
show_step = 2000
update_step = 100
batch_size = 100
pre_trained_step = 20000
tau = 0.0001
mainQN = QNetwork()
targetQN = QNetwork()
saver = tf.train.Saver(max_to_keep=3)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf_vars = tf.trainable_variables()
    update_target = update_target_graph(tf_vars, tau)
    sess.run(update_target)
    total_step = 0
    for episode in range(1, eposide_num + 1):
        start = time.time()
        episode_buffer = replay_buffer()
        observation = env.reset()
        done = False
        step = 0

        while not done:
            if total_step < pre_trained_step and np.random.rand() > greedy:
                action = np.random.randint(0, 4)
            else:
                action = mainQN.get_predict(sess, [observation])[0]

            observation_next, reward, done, max_num = env.step(action)
            step += 1
            total_step += 1
            if step > max_step:
                done = True

            episode_buffer.add((observation, action, reward, done, observation_next))
            if total_step > pre_trained_step:
                if total_step % update_step == 0:
                    train_buffer = gloal_buffer.sample(batch_size)
                    batch_observation_next = np.vstack(train_buffer[:, 4])
                    batch_actions = mainQN.get_predict(sess, batch_observation_next)
                    batch_target_Q_next = targetQN.get_Qout(sess, batch_observation_next)
                    doubleQ = batch_target_Q_next[range(batch_size), batch_actions]
                    targetQ = train_buffer[:, 2] + doubleQ * 0.99
                    mainQN.learn(sess, env_input=np.vstack(train_buffer[:, 0]), actions=batch_actions, targetQ=targetQ)
                    sess.run(update_target)

        gloal_buffer.add_all(episode_buffer.buffer)

        if episode % show_step == 0:
            saver.save(sess, 'model/dqn', global_step=episode, write_meta_graph=(episode == show_step))
            print(
                "episode = {} ,duration = {:.5f} , step = {} ,final_score = {} ".format(episode, time.time() - start,
                                                                                        step,
                                                                                        max_num))
