from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import tensorflow as tf


class Generator():
    def __init__(self, width, height, chars, image_len):
        self._generator = ImageCaptcha(width, height)
        self._chars = chars
        self._image_len = image_len
        self._char_classes = len(chars)

    def generate_label(self):
        label = ''.join(np.random.choice(self._chars, self._image_len, replace=False))
        return [label]

    def generate_image(self, label):
        if type(label) == bytes:
            label = label.decode()
        data = self._generator.generate(label)
        return np.asarray(Image.open(data)).transpose([1, 0, 2])

    def encode_label(self, label):
        if type(label) == bytes:
            label = label.decode()
        data = np.zeros([self._char_classes * self._image_len], dtype=np.float32)
        for i in range(self._image_len):
            data[i * self._char_classes + self._chars.index(label[i])] = 1
        return data

    def dataset(self, batch_size):
        dataset = tf.data.Dataset.from_generator(self.generate_label, output_types=tf.string)
        dataset = dataset.map(lambda label: (tf.py_func(func=self.generate_image, inp=[label], Tout=tf.uint8),
                                             tf.py_func(func=self.encode_label, inp=[label], Tout=tf.float32)))
        dataset = dataset.repeat().batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
