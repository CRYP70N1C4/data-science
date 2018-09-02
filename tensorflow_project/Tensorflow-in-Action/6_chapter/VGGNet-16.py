import tensorflow as tf
import tensorflow.contrib.slim as slim

def predict(images, keep_prob):
    conv1_1 = slim.conv2d(images, 64, 3)
    conv1_2 = slim.conv2d(conv1_1, 64, 3)
    pool1 = slim.max_pool2d(conv1_2, 2, 2, padding='SAME')

    conv2_1 = slim.conv2d(pool1, 256, 3)
    conv2_2 = slim.conv2d(conv2_1, 256, 3)
    pool2 = slim.max_pool2d(conv2_2, 2, 2, padding='SAME')

    conv3_1 = slim.conv2d(pool2, 256, 3)
    conv3_2 = slim.conv2d(conv3_1, 256, 3)
    conv3_3 = slim.conv2d(conv3_2, 256, 3)
    pool3 = slim.max_pool2d(conv3_3, 2, 2, padding='SAME')

    conv4_1 = slim.conv2d(pool3, 512, 3)
    conv4_2 = slim.conv2d(conv4_1, 512, 3)
    conv4_3 = slim.conv2d(conv4_2, 512, 3)
    pool4 = slim.max_pool2d(conv4_3, 2, 2, padding='SAME')

    conv5_1 = slim.conv2d(pool4, 512, 3)
    conv5_2 = slim.conv2d(conv5_1, 512, 3)
    conv5_3 = slim.conv2d(conv5_2, 512, 3)
    pool5 = slim.max_pool2d(conv5_3, 2, 2, padding='SAME')

    resh1 = slim.flatten(pool5)
    fc1 = slim.fully_connected(resh1, 4096, keep_prob=keep_prob)
    fc2 = slim.fully_connected(fc1, 4096, keep_prob=keep_prob)
    fc3 = slim.fully_connected(fc2, 1000, activation_fn=tf.nn.softmax)
    return fc3
