import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def get_next_batch(batch_size=128):
    images = []
    labels = []
    for _ in range(batch_size):
        label = np.random.randint(0, 10)
        image = np.random.randint(0, 255, [24, 24, 3])
        image[0] = np.ones([24, 3]) * label
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


def predict(x_input):
    f1 = slim.flatten(x_input)
    fl1 = slim.fully_connected(f1, 128)
    fl2 = slim.fully_connected(fl1, 10, activation_fn=tf.nn.sigmoid)
    return fl2


x_input = tf.placeholder(tf.float32, [None, 24, 24, 3])
y_input = tf.placeholder(tf.int64, [None])
logits = predict(x_input)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_input, tf.argmax(logits, -1)), tf.float32))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000001):
        batch_x, batch_y = get_next_batch()
        feed_dict = {x_input: batch_x, y_input: batch_y}
        sess.run(train_op, feed_dict=feed_dict)
        if step % 500 == 0:
            loss_val, accuracy_val = sess.run([loss, accuracy], feed_dict=feed_dict)
            print("step = {} , loss = {:.5f} , accuracy = {:.5f} ".format(step, loss_val, accuracy_val))
