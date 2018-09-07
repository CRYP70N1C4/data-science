import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat


# https://www.jianshu.com/p/c9fd5c01715e

def import_pd_to_tensorboard():
    with tf.Session() as sess:
        model_filename = 'model_dir/saved_model.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            if 1 != len(sm.meta_graphs):
                print('More than one graph found. Not sure which to write')
                sys.exit(1)

            tf.import_graph_def(sm.meta_graphs[0].graph_def)
    train_writer = tf.summary.FileWriter("log_dir")
    train_writer.add_graph(sess.graph)
    train_writer.close()


def get_all_tensor_names():
    with tf.Session() as sess:
        model_filename = 'model_dir/saved_model.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            if 1 != len(sm.meta_graphs):
                print('More than one graph found. Not sure which to write')
                sys.exit(1)

            tf.import_graph_def(sm.meta_graphs[0].graph_def)
            [print(n.name) for n in tf.get_default_graph().as_graph_def().node]


if __name__ == '__main__':
    get_all_tensor_names()
