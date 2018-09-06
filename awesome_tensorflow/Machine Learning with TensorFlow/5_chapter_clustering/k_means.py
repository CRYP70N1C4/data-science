import numpy as np
import tensorflow as tf

np.random.seed(0)

K = 5


def gen_data(batch_size=50):
    num = batch_size // K
    data = []
    for i in range(1, K + 1):
        mid = i * 100
        x = np.random.randint(mid - 10, mid + 10, size=[num, 2])
        data.append(x)
    data = np.vstack(data)
    np.random.shuffle(data)
    return data


def initial_cluster_centroids(X):
    return X[0:K, :]


def assign_cluster(X, centroids):
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    return mins


def recompute_centroids(X, Y):
    sums = tf.unsorted_segment_sum(X, Y, K)
    counts = tf.unsorted_segment_sum(tf.ones_like(X), Y, K)
    return sums / counts


x_input = tf.placeholder(tf.float32, [None, 2])
origin_center = tf.placeholder(tf.float32, [K, 2])
labels = assign_cluster(x_input, origin_center)
next_center = recompute_centroids(x_input, labels)

with tf.Session() as sess:
    x_input_val = gen_data()
    origin_center_val = initial_cluster_centroids(x_input_val)
    for i in range(100):
        feed_dict = {x_input: x_input_val, origin_center: origin_center_val}
        origin_center_val = sess.run(next_center, feed_dict=feed_dict)
    labels_val = sess.run(labels, feed_dict=feed_dict)

    for i in range(len(labels_val)):
        print(x_input_val[i], origin_center_val[labels_val[i]])
