import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.saved_model import tag_constants

graph = tf.Graph()
with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess,
        [tag_constants.SERVING],
        'model_dir',
    )
    batch_size_placeholder = graph.get_tensor_by_name('x:0')
    features_placeholder = graph.get_tensor_by_name('features_placeholder:0')
    labels_placeholder = graph.get_tensor_by_name('labels_placeholder:0')
