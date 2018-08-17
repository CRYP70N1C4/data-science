from game_2048 import Game
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import random


def discount_rewards(r, alpha=0.99):
    discount_r = np.zeros_like(r, dtype=np.float32)
    award = 0.
    for i in reversed(range(len(r))):
        award = r[i] + award * alpha
        discount_r[i] = award
    return discount_r


def predict(x_input):
    hidden_layer = slim.fully_connected(x_input, 50)
    return slim.fully_connected(hidden_layer, 4, activation_fn=tf.nn.sigmoid)


def get_loss(actions, probability, advantages):
    axis = tf.expand_dims(tf.range(tf.shape(actions)[0]), 1)
    ids = tf.concat([axis, actions], 1)
    prop = tf.gather_nd(probability, ids)
    return -tf.reduce_mean(prop * advantages)


observations = tf.placeholder(tf.float32, [None, 16])
actions = tf.placeholder(tf.int32, [None, 1])
advantages = tf.placeholder(tf.float32, [None, 1])
probability = predict(observations)
loss = get_loss(actions, probability, advantages)
tvars = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

W1Grad = tf.placeholder(tf.float32)
W2Grad = tf.placeholder(tf.float32)
newGrads = tf.gradients(loss, tvars)
updateGrads = optimizer.apply_gradients(zip([W1Grad, W2Grad], tvars))

xs, ys, drs = [], [], []
rendering = False
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rendering = False
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    game = Game()
    done = False
    episode_number = 0
    while episode_number < 2000000:
        observation = game.reset()
        done = False
        step = 0
        while not done:
            x = np.reshape(observation, [1, 16])
            tfprob = sess.run(probability, feed_dict={observations: x})
            action = np.argmax(tfprob)
            xs.append(x)
            ys.append(action)
            observation, space, reward, done, max_num = game.step(action)
            step += 1
            drs.append(reward)
            if max_num > 1024:
                rendering = True
            if done or step > 2000:
                done = True
                if rendering:
                    print(game)
                print(max_num)
                episode_number += 1
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                xs, ys, drs = [], [], []
                # 计算每一步潜在价值
                discounted_epr = discount_rewards(epr)
                # 得到的三个数据输入神经网络，返回求解梯度
                tGrad = sess.run(newGrads, feed_dict={observations: epx, actions: epy, advantages: discounted_epr})
                # 将获得的梯度累加到gradBuffer中
                for ix, grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

                if episode_number % 64 == 0:
                    sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                    for ix, grad in enumerate(tGrad):
                        gradBuffer[ix] = grad * 0
