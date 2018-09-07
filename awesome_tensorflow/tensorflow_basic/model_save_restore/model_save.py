import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

np.random.seed(10)


def inference(x_input):
    net = slim.fully_connected(x_input, 5)
    net = slim.fully_connected(net, 1, activation_fn=None)
    return net


def get_train_data(batch_size):
    batch_x = np.random.normal(10, 5, [batch_size, 3])
    batch_y = 0.4 * batch_x[:, 0] + 0.8 * batch_x[:, 1] + batch_x[:, 2] + np.random.uniform(-1, 1, batch_size)
    return batch_x, batch_y


x_input = tf.placeholder(tf.float32, [None, 3])
y_input = tf.placeholder(tf.float32, [None])
y_pred = inference(x_input)
loss = tf.reduce_mean(tf.square(tf.subtract(y_pred, tf.reshape(y_input, [-1, 1]))))
train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
tf.summary.scalar("loss", loss)
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log_dir/", graph=tf.get_default_graph())
    for i in range(1, 10001):
        batch_x, batch_y = get_train_data(32)
        feed_dict = {x_input: batch_x, y_input: batch_y}
        sess.run(train_op, feed_dict=feed_dict)
        if i % 200 == 0:
            print(i)
            summary = sess.run(merged_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)
    tf.saved_model.simple_save(sess, export_dir="model_dir", inputs={"x": x_input}, outputs={"y": y_pred})
