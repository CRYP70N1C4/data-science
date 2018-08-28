import tensorflow as tf
from tensorflow.contrib import slim
from captcha.image import ImageCaptcha
import string
from PIL import Image
import numpy as np

image_generator = ImageCaptcha(width=120, height=50)
chars = [ch for ch in (string.ascii_lowercase + string.digits)]
char_classes = len(chars)
image_len = 4
batch_size = 64
epoch = 32


def generate_label():
    label = ''.join(np.random.choice(chars, image_len, replace=False))
    return [label]


def generate_image(label, gray=False):
    # convert('L')
    label = label.decode()
    data = image_generator.generate(label)
    return np.asarray(Image.open(data)).transpose([1, 0, 2])


def encode_label(label):
    label = label.decode()
    data = np.zeros([char_classes * image_len], dtype=np.float32)
    for i in range(image_len):
        data[i * char_classes + chars.index(label[i])] = 1
    return data


def get_train_data():
    dataset = tf.data.Dataset.from_generator(generate_label, output_types=tf.string)
    dataset = dataset.map(lambda label: {'labels': tf.py_func(func=encode_label, inp=[label], Tout=tf.float32),
                                         'images': tf.py_func(func=generate_image, inp=[label], Tout=tf.uint8)})
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator


def predict(x_input, keep_prob):
    net = tf.reshape(x_input, [-1, 120, 50, 3])
    net = slim.conv2d(net, 32, [5, 5])
    net = slim.conv2d(net, 64, [5, 5])
    net = slim.flatten(net)
    net = slim.fully_connected(net, 1024)
    net = slim.dropout(net, keep_prob=keep_prob)
    net = slim.fully_connected(net, char_classes * image_len, activation_fn=None)
    return net


def get_loss(y_input, y_pred):
    y_expect_reshaped = tf.reshape(y_input, [-1, image_len, char_classes])
    y_got_reshaped = tf.reshape(y_pred, [-1, image_len, char_classes])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_expect_reshaped, logits=y_got_reshaped))
    return cross_entropy


x_input = tf.placeholder(tf.float32, [None, 120, 50, 3])
y_input = tf.placeholder(tf.float32, [None, char_classes * image_len])
keep_prob = tf.placeholder(tf.float32)
y_pred = predict(x_input, keep_prob)
loss = get_loss(y_input, y_pred)
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

iterator = get_train_data()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    for i in range(1, 10000):
        train_data = sess.run(next_element)
        labels = train_data['labels']
        images = train_data['images']
        feed_dict = {x_input: images, y_input: labels, keep_prob: 0.75}
        sess.run(train_op, feed_dict=feed_dict)
        if i % 50 == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            print("step = {} ,loss = {:.5f}".format(i, loss_val))
