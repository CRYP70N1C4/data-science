import cifar10_input
import numpy as np
import time
import tensorflow as tf

images_train, labels_train = cifar10_input.distorted_inputs(data_dir='../../../_dataset/cifar-10-batches-py',
                                                            batch_size=128)

images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir='../../../_dataset/cifar-10-batches-py',
                                                batch_size=128)

x_input = tf.placeholder(tf.float32, [None, 24, 24, 3])
y_input = tf.placeholder(tf.float32, [None])


def predict(x_input):
    conv1 = tf.layers.conv2d(x_input, 64, 5, padding='SAME', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 3, 2, padding='SAME')
    norm1 = tf.nn.lrn(pool1, 4, bias=1, alpha=0.001 / 9.0, beta=0.75)

    conv2 = tf.layers.conv2d(norm1, 64, 5, padding='SAME', activation=tf.nn.relu)
    norm2 = tf.nn.lrn(conv2, 4, bias=1, alpha=0.001 / 9.0, beta=0.75)
    pool2 = tf.layers.max_pooling2d(norm2, 3, 2, padding='SAME')
    flatten1 = tf.layers.flatten(pool2)
    layer1 = tf.layers.dense(flatten1, 384, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 192, activation=tf.nn.relu)
    layer3 = tf.layers.dense(layer2, 10)
    return layer3


def get_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    tf.add_to_collection("losses", cross_entropy_mean)
    return tf.add_n(tf.get_collection("losses"), name='totoal_loss')


def get_accuracy(logits, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits), tf.cast(labels, tf.int64)), tf.float32))


logits = predict(x_input)
loss = get_loss(logits, y_input)

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
accuracy = get_accuracy(logits, y_input)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()
    for step in range(1, 30001):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        sess.run(train_op, feed_dict={x_input: image_batch, y_input: label_batch})
        duration = time.time() - start_time

        if step % 50 == 0:
            cur_loss, cur_accuracy = sess.run([loss, accuracy], feed_dict={x_input: image_batch, y_input: label_batch})
            print("step = {}, time cost = {} ,loss = {:.5f},accuracy = {:.5f}".format(step, duration, cur_loss,
                                                                                      cur_accuracy))
