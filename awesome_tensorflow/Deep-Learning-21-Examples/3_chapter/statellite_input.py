import tensorflow as tf
from tensorflow.contrib import slim
import os

SPLITS_TO_SIZES = {'train': 3320, 'validation': 350}


def get_split(split_name, dataset_dir='statellite', file_pattern='statellite_%s_*.tfrecord'):
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/text': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'label_text': slim.tfexample_decoder.Tensor('image/class/text')}

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions={},
        num_classes=6,
        labels_to_names=None)


dataset = get_split('train')

provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

[image, label] = provider.get(['image', 'label'])

sess = tf.Session()
print(sess.run(label))