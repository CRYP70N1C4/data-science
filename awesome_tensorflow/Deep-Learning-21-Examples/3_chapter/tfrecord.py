# -*- coding:utf8 -*-

import tensorflow as tf
import os
import numpy as np
import threading

np.random.seed(0)


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_record(image_buffer, label, text):
    channels = 3
    height = 256
    width = 256
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(text.encode()),
        'image/format': _bytes_feature(image_format.encode()),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()
        self.path = tf.placeholder(dtype=tf.string)
        self.img_data = tf.read_file(self.path)

    def read_data(self, img_path):
        image = self._sess.run(self.img_data,
                               feed_dict={self.path: img_path})
        return image


def _process_image_files_batch(coder: ImageCoder, filenames, thread_index, ranges, tag,
                               texts, labels, num_shards, dataset_name, output_dir):
    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards + 1).astype(int)
    total_shards = int(num_shards / len(ranges))
    for i in range(total_shards):
        output_filename = '%s_%s_%.5d.tfrecord' % (dataset_name, tag, total_shards * thread_index + i)
        print(output_filename)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        for j in range(shard_ranges[i], shard_ranges[i + 1]):
            label = labels[i]
            text = texts[i]
            image = coder.read_data(filenames[i])
            record = _convert_to_record(image, label, text)
            writer.write(record.SerializeToString())
        writer.close()


def _process_image_files(tag, filenames, texts, labels, num_threads, num_shards, dataset_name, output_dir):
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    coord = tf.train.Coordinator()

    coder = ImageCoder()
    threads = []
    for thread_index in range(num_threads):
        args = (coder, filenames, thread_index, ranges, tag,
                texts, labels, num_shards, dataset_name, output_dir)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    coord.join(threads)


def _find_image_files(data_dir, labels_file):
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]

    labels = []
    filenames = []
    texts = []
    label_index = 0
    for label in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir, label)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([label] * len(matching_files))
        filenames.extend(matching_files)
        label_index += 1

    shuffled_index = np.arange(len(filenames))
    np.random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    return filenames, texts, labels


def process_dataset(tag, directory, num_shards, labels_file):
    filenames, texts, labels = _find_image_files(directory, labels_file)
    _process_image_files(tag, filenames, texts, labels, 5, num_shards, "statellite", "statellite")
