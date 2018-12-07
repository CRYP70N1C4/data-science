import tensorflow as tf
from tensorflow.contrib import slim
from images_generator import Generator
import string


def get_accuracy(predict, labels):
    predict = tf.reshape(predict, [-1, 36, 4])
    predict = tf.argmax(predict, -1)
    labels = tf.reshape(labels, [-1, 36, 4])
    labels = tf.argmax(labels, -1)
    return tf.reduce_mean(tf.cast(tf.equal(predict, labels), tf.float32))


def vgg16(inputs, num_classes):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.flatten(net)
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, num_classes, activation_fn=tf.nn.sigmoid, scope='fc8')
    return net


chars = [ch for ch in (string.ascii_lowercase + string.digits)]
datasets = Generator(120, 50, chars, 4).dataset(32)
inputs = tf.placeholder(tf.float32, [None, 120, 50, 3])
labels = tf.placeholder(tf.float32, [None, len(chars) * 4])
predictions = vgg16(inputs, num_classes=len(chars) * 4)

slim.losses.softmax_cross_entropy(predictions, labels)

total_loss = slim.losses.get_total_loss()
accuracy = get_accuracy(predictions, labels)
tf.summary.scalar('losses/total_loss', total_loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)

train_op = slim.learning.create_train_op(total_loss, optimizer)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, 100001):
        batch_images, batch_labels = sess.run(datasets)
        feed_dict = {inputs: batch_images, labels: batch_labels}
        sess.run(train_op, feed_dict=feed_dict)
        if i % 100 == 0:
            loss_val, accuracy_val = sess.run([total_loss, accuracy], feed_dict=feed_dict)
            print("step = {} ,loss = {:.5f},accuracy = {:.5f}".format(i, loss_val, accuracy_val))
