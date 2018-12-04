import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import cifar_10


def inference(x_input):
    net = slim.conv2d(x_input, 64, [5, 5])
    net = slim.max_pool2d(net, kernel_size=[3, 3], padding='SAME')
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9, beta=0.75)

    net = slim.conv2d(net, 64, [5, 5])
    net = slim.max_pool2d(net, kernel_size=[3, 3], padding='SAME')
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9, beta=0.75)

    net = slim.flatten(net)
    net = slim.fully_connected(net, 384)
    net = slim.fully_connected(net, 192)
    net = slim.fully_connected(net, 10, activation_fn=tf.nn.sigmoid)
    return net


def get_loss_and_accuracy(labels, logits):
    labels = tf.cast(labels, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    accuracy = tf.equal(labels, tf.argmax(logits, -1))
    return tf.reduce_mean(loss), tf.reduce_mean(tf.cast(accuracy, tf.float32))


data_dir = "../../../_dataset/cifar-10-batches-bin"
batch_size = 64
episode_num = 100000
summary_step = 100

images = tf.placeholder(tf.float32, [None, 24, 24, 3])
labels = tf.placeholder(tf.float32)
logits = inference(images)
loss, accuracy = get_loss_and_accuracy(labels, logits)
train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
train_batch = cifar_10.load_data(data_dir, batch_size=batch_size)

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log_dir/", graph=tf.get_default_graph())
    for i in range(1, episode_num + 1):
        train_images, train_label = sess.run(train_batch)
        feed_dict = {images: train_images, labels: np.reshape(train_label, -1)}
        sess.run(train_op, feed_dict=feed_dict)
        if i % summary_step == 0:
            loss_val, accuracy_val, summary = sess.run([loss, accuracy, merged_summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)
            print("step = {},accuracy = {:.5f}, loss = {:.5f}".format(i, accuracy_val, loss_val))
