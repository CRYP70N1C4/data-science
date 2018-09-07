import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow


def desc_model(checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))


def model_save():
    x_input = tf.placeholder(tf.float32, [None, 3], name='x_input')
    y_input = tf.placeholder(tf.float32, [None])

    layer1 = tf.layers.dense(x_input, 10)
    y_pred = tf.layers.dense(layer1, 1, name='y_pred')

    loss = tf.reduce_mean((y_pred - y_input) ** 2)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    saver = tf.train.Saver(max_to_keep=4)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, 1000001):
            batch_x = np.random.randint(-10, 10, [32, 3])
            batch_y = 0.4 * batch_x[:, 0] + 0.8 * batch_x[:, 1] + batch_x[:, 2] + np.random.uniform(-1, 1, 32)
            sess.run(train_op, feed_dict={x_input: batch_x, y_input: batch_y})

            if epoch % 1000 == 0:
                cur_loss = sess.run(loss, feed_dict={x_input: batch_x, y_input: batch_y})
                print("step = {} ,loss = {:.5f} ".format(epoch, cur_loss))
                saver.save(sess, "test_model/my-model", global_step=epoch)

        tmp = sess.run(y_pred, feed_dict={x_input: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])})
        print(tmp)


def get_by_name(graph, name):
    try:
        return graph.get_operation_by_name(name).outputs[0]
    except Exception:
        return graph.get_tensor_by_name(name)


def model_restore():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('test_model/my-model-1000000.meta')
        saver.restore(sess, tf.train.latest_checkpoint("test_model/"))
        graph = tf.get_default_graph()
        nodes = [n.name for n in graph.as_graph_def().node]
        x_input = graph.get_operation_by_name("x_input").outputs[0]
        feed_dict = {x_input: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}
        for node in nodes:
            try:
                y_pred = get_by_name(graph, node)
                if y_pred.get_shape().as_list() == [None, 1]:
                    tmp = sess.run(y_pred, feed_dict=feed_dict)
                    print("\n", node)
                    print(tmp)
            except Exception as e:
                pass


if __name__ == '__main__':
    model_save()
    model_restore()
