import tensorflow as tf
import os
from PIL import Image
import numpy as np

img_width = 32
img_height = 32
img_depth = 3
label_bytes = 1
image_bytes = img_width * img_height * img_depth


def parse_record(value):
    record_bytes = tf.decode_raw(value, tf.uint8)
    label = tf.slice(record_bytes, [0], [label_bytes])
    label = tf.cast(label, tf.int32)
    image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
    image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
    image = tf.transpose(image_raw, (1, 2, 0))  # convert from D/H/W to H/W/D
    image = tf.cast(image, tf.float32)
    # 数据增强
    image = tf.random_crop(image, [24, 24, 3])  # 随机剪裁图片 从32*32*3 到 24*24*3
    image = tf.image.random_flip_left_right(image)  # 0.5的概率左右旋转
    image = tf.image.random_brightness(image, max_delta=63)  # 随机改变亮度
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)  # 随机改变对比度
    return image, label


def load_data(data_dir, train=True, batch_size=64):
    if train:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]

    dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes=image_bytes + label_bytes)
    dataset = dataset.map(parse_record)
    iterator = dataset.repeat().batch(batch_size).shuffle(buffer_size=2000).make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element


def sample_image(size=1):
    data_dir = "../../../_dataset/cifar-10-batches-bin"
    train_batch = load_data(data_dir, batch_size=size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        images, label = sess.run(train_batch)
        for i in range(size):
            Image.fromarray(images[i].astype(np.uint8)).show()
            print(label[i])

