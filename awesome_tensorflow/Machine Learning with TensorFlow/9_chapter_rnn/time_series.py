import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn

np.random.seed(0)


def gen_data(x):
    def fake(x):
        x = 2 * x + 100
        return 10 * (x // 5) - 10 * (x % 5) ** 2 + np.random.rand()

    return x, np.array(list(map(fake, x)))


def init(sess, model_dir):
    try:
        tf.get_variable_scope().reuse_variables()
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(model_dir))
        print("init from file success")
    except Exception as ex:
        print(ex)
        sess.run(tf.global_variables_initializer())


class SeriesPredictor():

    def __init__(self, input_dim, seq_size):
        self.input_dim = input_dim
        self.seq_size = seq_size

        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])
        self.y_pred = self.inference()

        self.loss = tf.reduce_mean(tf.square(self.y_pred - self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=3)

    def inference(self):
        cell = rnn.BasicLSTMCell(20)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        W_out = tf.Variable(tf.random_normal([20, 1]), name='W_out')
        b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        W_repeated = tf.tile(tf.expand_dims(W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            init(sess, 'model/')
            for i in range(10000):
                feed_dict = {self.x: train_x, self.y: train_y}
                sess.run(self.train_op, feed_dict=feed_dict)
                if i % 1000 == 0:
                    loss_val = sess.run(self.loss, feed_dict=feed_dict)
                    print("step = {} ,loss = {:.5f}".format(i, loss_val))
            self.saver.save(sess, 'model/series')

    def predict(self, x):
        with tf.Session() as sess:
            init(sess, 'model/')
            y_pred_val = sess.run(self.y_pred, feed_dict={self.x: x})
            return y_pred_val


def train():
    x = np.arange(1000, 1100)
    x, y = gen_data(x)
    x = np.reshape(x, [-1, 5, 1])
    y = np.reshape(y, [-1, 5])
    model = SeriesPredictor(input_dim=1, seq_size=5)
    model.train(x, y)


def test():
    x, y = gen_data(100)
    x1 = x[-50:]
    x1 = np.reshape(x1, [-1, 5, 1])
    model = SeriesPredictor(input_dim=1, seq_size=5)
    y_pred = model.predict(x1)
    x1 = np.reshape(x1, [-1])
    y_pred = np.reshape(y_pred, [-1])

    plt.plot(x, y, 'r*', x1, y_pred, 'g+')
    plt.show()


if __name__ == '__main__':
    train()
    # test()
