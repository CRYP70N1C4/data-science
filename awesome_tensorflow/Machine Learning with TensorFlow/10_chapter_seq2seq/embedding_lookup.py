import tensorflow as tf

sess = tf.InteractiveSession()

embeddings_0d = tf.constant([17, 22, 35, 51])

embeddings_4d = tf.constant([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

embeddings_2x2d = tf.constant([[[1, 0], [0, 0]],
                               [[0, 1], [0, 0]],
                               [[0, 0], [1, 0]],
                               [[0, 0], [0, 1]]])

ids = tf.constant([1, 0, 2])

lookup_0d = sess.run(tf.nn.embedding_lookup(embeddings_0d, ids))
lookup_1d = sess.run(tf.nn.embedding_lookup(embeddings_4d, ids))
lookup_2d = sess.run(tf.nn.embedding_lookup(embeddings_2x2d, ids))

print(lookup_0d)
print(lookup_1d)
print(lookup_2d)
