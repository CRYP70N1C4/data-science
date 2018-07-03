import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(low=low, high=-low, size=[fan_in, fan_out]).astype(np.float32)


def predict(x_input):
    hidden = tf.layers.dense(x_input, 200,
                             kernel_initializer=tf.constant_initializer(xavier_init(784, 200)))
    out = tf.layers.dense(hidden, 784)
    return out


X = tf.placeholder(tf.float32, [None, 784])
Y = predict(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=X, logits=Y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

mnist = input_data.read_data_sets("../../../_dataset/MNIST_data", one_hot=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_X, _ = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={X: batch_X})
        if i % 10 == 0:
            print("step = {} ,loss = {:.5f}".format(i, sess.run(loss, feed_dict={X: batch_X})))
