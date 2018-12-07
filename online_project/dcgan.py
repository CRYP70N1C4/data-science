import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

batch_size = 64
NUM_CLASSES = 32
WIDTH = 128
HEIGHT = 128


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def discriminator(image, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=lrelu):
            net = image
            f = 32
            for i in range(5):
                if i < 3:
                    net = slim.conv2d(net, f, 4, 2)
                else:
                    net = slim.conv2d(net, f, 3, 2)
                net = lrelu(slim.conv2d(slim.conv2d(net, f, 3, 1), f, 3, 1, activation_fn=None) + net)
                net = lrelu(slim.conv2d(slim.conv2d(net, f, 3, 1), f, 3, 1, activation_fn=None) + net)

                f *= 2
            net = slim.conv2d(net, f, 3, 2)
            net = slim.flatten(net)
            return slim.fully_connected(net, NUM_CLASSES), slim.fully_connected(net, 1)


def generator(noise,label):
    with tf.variable_scope("discriminator"):
        pass

# images = tf.placeholder(dtype=tf.float32, shape=[batch_size, HEIGHT, WIDTH, 3])
# x, y = discriminator(images)
# print(x)
# print(y)
