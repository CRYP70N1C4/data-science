import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../../_dataset/MNIST_data", one_hot=True)

x_input = tf.placeholder(tf.float32, [None, 784])
y_input = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


def predict(net):
    layer1 = tf.layers.dense(net, 300, activation=tf.nn.relu)
    dropout = tf.nn.dropout(layer1, keep_prob=keep_prob)
    out = tf.layers.dense(dropout, 10, activation=tf.nn.sigmoid)
    return out


logits = predict(x_input)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_input))
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_input, -1), tf.argmax(logits, -1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1, 3001):
        batch_x, batch_y = mnist.train.next_batch(128)
        sess.run(train_op, feed_dict={x_input: batch_x, y_input: batch_y, keep_prob: 0.75})
        if i % 100 == 0:
            cur_loss, cur_accuracy = sess.run([loss, accuracy],
                                              feed_dict={x_input: batch_x, y_input: batch_y, keep_prob: 0.75})
            print("step = {} ,loss = {:.5f} ,accuracy = {:.5f}".format(i, cur_loss, cur_accuracy))

    batch_x, batch_y = mnist.test.next_batch(128)
    cur_loss, cur_accuracy = sess.run([loss, accuracy], feed_dict={x_input: batch_x, y_input: batch_y, keep_prob: 1.0})
    print("test loss = {:.5f} ,accuracy = {:.5f}".format(cur_loss, cur_accuracy))
