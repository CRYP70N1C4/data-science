import tensorflow as tf
import numpy as np

np.random.seed(0)
batch_size = 32
episode_num = 40000


def get_train_data(batch_size=batch_size):
    x = np.random.normal(100, 20, size=[batch_size, 4])
    y = np.sum(np.multiply([[0.1, 0.2, 0.4, 0.8]], x), axis=-1) + 1 + 0.01 * np.random.rand()
    return x, y


def predict(x_input):
    W = tf.Variable(np.random.random(size=4), dtype=np.float32, name="weight")
    b = tf.Variable(np.random.random(), dtype=np.float32, name="bias")
    return tf.reduce_sum(x_input * W, -1) + b


x_input = tf.placeholder(tf.float32, [None, 4])
y_input = tf.placeholder(tf.float32)
y_pred = predict(x_input)
loss = tf.reduce_mean((y_input - y_pred) ** 2)
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

tf.summary.scalar("loss", loss)
tf.summary.histogram("loss", y_pred)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log_dir/", graph=tf.get_default_graph())
    for i in range(1, episode_num + 1):
        train_x, train_y = get_train_data()
        feed_dict = {x_input: train_x, y_input: train_y}
        sess.run(train_op, feed_dict=feed_dict)
        if i % 500 == 0:
            loss_val, summary = sess.run([loss, merged_summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)
            print("step = {}, loss = {:.5f}".format(i, loss_val))

    summary_writer.close()
    b_val, w_val = sess.run(
        [tf.get_default_graph().get_tensor_by_name('bias:0'), tf.get_default_graph().get_tensor_by_name('weight:0')])
    print(b_val)
    print(w_val)
    print('tensorboard --logdir=log_dir')
