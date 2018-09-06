import tensorflow as tf


def convert_png_to_jpg(png_path, jpg_path):
    _png_path = tf.placeholder(tf.string)
    png_data = tf.read_file(_png_path)
    image = tf.image.decode_png(png_data, channels=3)
    jpg_image = tf.image.encode_jpeg(image, format='rgb', quality=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img = sess.run(jpg_image, feed_dict={_png_path: png_path})
        with open(jpg_path, 'wb') as f:
            f.write(img)


def _unsorted_segment_sum():
    c = tf.constant([[1, 1], [2, 2], [3, 3], [4, 4]])
    result = tf.unsorted_segment_sum(c, tf.constant([1, 0, 1, 2]), 3)
    sess = tf.Session()
    print(sess.run(result))


def _slice():
    x = tf.constant([[[1, 1, 1], [2, 2, 2]],
                     [[3, 3, 3], [4, 4, 4]],
                     [[5, 5, 5], [6, 6, 6]]])

    y = tf.slice(x, [1, 0, 0], [2, 1, 3])

    sess = tf.Session()
    print(sess.run(y))


def _gather():
    mapping_strings = tf.constant(["emerson", "lake", "palmer"])
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings, num_oov_buckets=1, default_value=-1)
    features = tf.constant([["emerson"], ["palmer"]])
    ids = table.lookup(features)
    select_features = tf.gather(mapping_strings, ids)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print(sess.run(ids))
        print(sess.run(select_features).astype(np.str))
