import string
import tensorflow as tf
from images_generator import Generator
from vgg16 import vgg16
from cnn import cnn


def get_model(name):
    if name == "vgg16":
        return vgg16(120, 50, 36, 4)
    if name == "cnn":
        return cnn(120, 50, 36, 4)


def init(sess, model_dir):
    try:
        tf.get_variable_scope().reuse_variables()
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(model_dir))
        print("init from %s success" % model_dir)
    except Exception as ex:
        print("init from %s fail" % model_dir)
        print(ex)
        sess.run(tf.global_variables_initializer())


chars = [ch for ch in (string.ascii_lowercase + string.digits)]
datasets = Generator(120, 50, chars, 4).dataset(32)
model_name = "cnn"
model = get_model(model_name)
saver = tf.train.Saver(max_to_keep=3)
with tf.Session() as sess:
    init(sess, model_name)
    for step in range(1, 100001):
        images, labels = sess.run(datasets)
        if step % 100 == 0:
            [loss, accuracy, _] = model.train(sess, images, labels, True)[0]
            print("step = {} ,loss = {:.5f} ,accuracy = {:.5f} ".format(step, loss, accuracy))
        else:
            model.train(sess, images, labels, False)
        if step % 2000 == 0:
            saver.save(sess, "%s/" % model_name)
