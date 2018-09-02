import tensorflow as tf


def predict(images):
    conv1 = tf.layers.conv2d(images, 64, 11, 4, padding='SAME', activation=tf.nn.relu)
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.0001, beta=0.75)
    pool1 = tf.layers.max_pooling2d(lrn1, 3, 2, padding='valid')

    conv2 = tf.layers.conv2d(pool1, 192, 5, padding='SAME', activation=tf.nn.relu)
    lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001, beta=0.75)
    pool2 = tf.layers.max_pooling2d(lrn2, 3, 2, padding='valid')

    conv3 = tf.layers.conv2d(pool2, 384, 3, padding='SAME', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3, 256, 3, padding='SAME', activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv4, 256, 3, padding='SAME', activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv5, 3, 2, padding='valid')

    fc1 = tf.layers.dense(pool3, 4096, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 4096, activation=tf.nn.relu)
    fc3 = tf.layers.dense(fc2, 1000)
    return fc3
