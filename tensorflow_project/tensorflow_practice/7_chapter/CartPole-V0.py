import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym


def discount_rewards(r):
    discount_r = np.zeros_like(r)
    tmp = 0;
    for i in reversed(range(len(r))):
        tmp = tmp * 0.99 + r[i]
        discount_r[i] = tmp
    return discount_r


def predict(x_input):
    hidden_layer = slim.fully_connected(x_input, 50)
    return slim.fully_connected(hidden_layer, 2, activation_fn=tf.nn.sigmoid)


def get_loss(actions, probability, advantages):
    axis = tf.expand_dims(tf.range(tf.shape(actions)[0]), 1)
    ids = tf.concat([axis, actions], 1)
    prop = tf.gather_nd(probability, ids)
    print(prop,advantages)
    return -tf.reduce_mean(prop * advantages)


def clear_buffer(buffer):
    for ix, data in enumerate(buffer):
        buffer[ix] = data * 0
    return buffer


observations = tf.placeholder(tf.float32, [None, 4])
actions = tf.placeholder(tf.int32, [None, 1])
advantages = tf.placeholder(tf.float32, [None, 1])
probability = predict(observations)
loss = get_loss(actions, probability, observations)

tvars = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
W1Grad = tf.placeholder(tf.float32)
W2Grad = tf.placeholder(tf.float32)

newGrads = tf.gradients(loss, tvars)
updateGrads = optimizer.apply_gradients(zip([W1Grad, W2Grad], tvars))

xs, ys, drs = [], [], []
with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    sess.run(tf.global_variables_initializer())
    gradBuffer = sess.run(tvars)
    gradBuffer = clear_buffer(gradBuffer)
    env = gym.make('CartPole-v0')
    observation = env.reset()
    episode_num = 0
    reward_sum = 0
    while episode_num <= 20000:
        x = np.reshape(observation, [1, 4])
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = np.argmax(tfprob)
        xs.append(x)
        ys.append(action)
        observation, reward, done, info = env.step(action)
        drs.append(reward)
        reward_sum += reward

        if done:
            episode_num += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(discount_rewards(drs))
            xs, ys, drs = [], [], []
            observation = env.reset()
            tGrads = sess.run(newGrads, feed_dict={observations: epx, actions: epy, advantages: epr})
            for ix, grad in tGrads:
                gradBuffer[ix] += grad

            if episode_num % 128 == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                gradBuffer = clear_buffer(gradBuffer)
                print(reward_sum / 128)
                reward_sum = 0
