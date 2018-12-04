import tensorflow as tf
from tensorflow.contrib import slim


class vgg16():
    def __init__(self, width, height, num_classes, char_len, channels=3):
        self.inputs = tf.placeholder(tf.float32, [None, width, height, channels])
        self.labels = tf.placeholder(tf.float32, [None, num_classes * char_len])
        self.num_classes = num_classes
        self.char_len = char_len
        self.predict = self._inference()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.predict))
        self.accuracy = self._get_accuracy()
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self.loss)

    def _inference(self):
        net = slim.conv2d(self.inputs, 64, [3, 3])
        net = slim.conv2d(net, 64, [3, 3])
        net = slim.max_pool2d(net, [2, 2], padding='SAME')

        net = slim.conv2d(net, 128, [3, 3])
        net = slim.conv2d(net, 128, [3, 3])
        net = slim.max_pool2d(net, [2, 2], padding='SAME')

        net = slim.conv2d(net, 256, [3, 3])
        net = slim.conv2d(net, 256, [3, 3])
        net = slim.max_pool2d(net, [2, 2], padding='SAME')

        net = slim.conv2d(net, 512, [3, 3])
        net = slim.conv2d(net, 512, [3, 3])
        net = slim.conv2d(net, 512, [3, 3])
        net = slim.max_pool2d(net, [2, 2], padding='SAME')

        net = slim.conv2d(net, 512, [3, 3])
        net = slim.conv2d(net, 512, [3, 3])
        net = slim.conv2d(net, 512, [3, 3])
        net = slim.max_pool2d(net, [2, 2], padding='SAME')

        net = slim.flatten(net)
        net = slim.fully_connected(net, 4096)
        net = slim.fully_connected(net, 4096)
        net = slim.fully_connected(net, self.char_len * self.num_classes, activation_fn=tf.nn.sigmoid)
        return net

    def train(self, sess, images, labels, loss_accuracy=False):
        ops = []
        if loss_accuracy:
            ops.append(self.loss)
            ops.append(self.accuracy)
        ops.append(self.train_op)
        return sess.run([ops], feed_dict={self.inputs: images, self.labels: labels})

    def _get_accuracy(self):
        predict = tf.reshape(self.predict, [-1, self.char_len, self.num_classes])
        predict = tf.argmax(predict, -1)
        labels = tf.reshape(self.labels, [-1, self.char_len, self.num_classes])
        labels = tf.argmax(labels, -1)
        return tf.reduce_mean(tf.cast(tf.equal(predict, labels), tf.float32))
