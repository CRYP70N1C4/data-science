import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


def get_train_data(batch_size=128):
    iris = datasets.load_iris()
    idx = np.random.choice(np.shape(iris['data'])[0], batch_size)
    features = iris['data'][idx]
    labels = iris['target'][idx]
    labels = np.array([1 if y == 0 else -1 for y in labels])
    return features[:, [0, 2]], np.reshape(labels, [-1, 1])


input_dim = 2
output_dim = 1

features = tf.placeholder(tf.float32, [None, input_dim])
labels = tf.placeholder(tf.float32, [None, output_dim])

A = tf.Variable(tf.random_normal(shape=[input_dim]))
b = tf.Variable(tf.random_normal(shape=[output_dim]))
pred = features * A - b

l2_norm = tf.reduce_sum(tf.square(A))

loss = tf.reduce_sum(tf.maximum(0., 1 - labels * pred)) + 0.01 * l2_norm
prediction = tf.sign(pred)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, 10001):
        batch_features, batch_labels = get_train_data()
        feed_dict = {features: batch_features, labels: batch_labels}
        sess.run(train_op, feed_dict=feed_dict)
        if step % 1000 == 0:
            loss_val, accuracy_val = sess.run([loss, accuracy], feed_dict=feed_dict)
            print("step = {},loss = {:.5f} , accuracy = {:.5f}".format(step, loss_val, accuracy_val))

    A_val, b_val = sess.run([A, b])
    print(A_val, b_val)

    batch_features, batch_labels = get_train_data()
    # Separate data
    setosa_x = [d[1] for i, d in enumerate(batch_features) if batch_labels[i][0] == 1]
    setosa_y = [d[0] for i, d in enumerate(batch_features) if batch_labels[i][0] == 1]
    not_setosa_x = [d[1] for i, d in enumerate(batch_features) if batch_labels[i][0] == -1]
    not_setosa_y = [d[0] for i, d in enumerate(batch_features) if batch_labels[i][0] == -1]

    f1 = []
    f2 = []
    for row in batch_features:
        x = row[1]
        y = (b_val[0] - A_val[1] * x) / A_val[0]
        f1.append(x)
        f2.append(y)

    plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
    plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
    plt.plot(f1, f2, 'r-', label='Linear Separator', linewidth=3)
    plt.ylim([0, 10])
    plt.legend(loc='lower right')
    plt.title('Sepal Length vs Petal Width')
    plt.xlabel('Petal Width')
    plt.ylabel('Sepal Length')
    plt.show()
