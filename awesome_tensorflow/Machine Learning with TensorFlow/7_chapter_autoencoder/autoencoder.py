import tensorflow as tf
from sklearn import datasets
import numpy as np

np.random.seed(0)


def get_batch(X, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a]


class AutoEncoder():

    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, 4])
        self.encode = self._encode()
        self.decode = self._decode()
        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.input, self.decode)))
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=3)

    def _encode(self):
        return tf.layers.dense(self.input, 2)

    def _decode(self):
        return tf.layers.dense(self.input, 4)

    def train(self, data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1, 20001):
                sess.run(self.train_op, feed_dict={self.input: get_batch(data, 100)})
                if i % 500 == 0:
                    loss_val = sess.run(self.loss, feed_dict={self.input: get_batch(data, 100)})
                    print("step = {},loss = {:.5f}".format(i, loss_val))
                    self.saver.save(sess, 'model/enocder', global_step=i, write_meta_graph=False)
            self.saver.save(sess, 'model/enocder', global_step=i)

    def test(self, data):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('model/model.ckpt-20000.meta')
            saver.restore(sess, tf.train.latest_checkpoint("model/"))
            hidden, reconstructed = sess.run([self.encode, self.decode], feed_dict={self.input: data})
        print('input', data)
        print('compressed', hidden)
        print('reconstructed', reconstructed)


data = datasets.load_iris().data

encoder = AutoEncoder()
# encoder.train(data)
encoder.test(data[0:5, ])
