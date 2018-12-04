import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

data_dir = 'D:\\迅雷下载\\DatasetA\\DatasetA';
train_img_dir = 'D:\\迅雷下载\\DatasetA\\DatasetA\\train'


def encode_label():
    label_dict = {}
    with open(os.path.join(data_dir, 'label_list.txt')) as f:
        for line in f.readlines():
            label = line.split('\t', 2)[0]
            label_dict[label] = len(label_dict)
    return label_dict


label_dict = encode_label()
print(len(label_dict))


def read_all_data():
    with open(os.path.join(data_dir, 'train.txt')) as f:
        return f.readlines()


train_data = read_all_data()
index = 0


def get_next_batch(batch_size=32):
    global index
    begin = index
    index = (index + batch_x) % len(train_data)
    for line in train_data[begin, index]:
        img_path = train_data[0]
        label = label_dict[train_data[1]]


def predict(x_input):
    conv1 = slim.conv2d(x_input, 256, [3, 3])
    conv2 = slim.conv2d(conv1, 512, [3, 3])
    conv3 = slim.conv2d(conv2, conv2, [3, 3])
    f1 = slim.flatten(conv3)
    fl1 = slim.fully_connected(f1, 512)
    fl2 = slim.fully_connected(fl1, 256)
    return slim.fully_connected(fl2, 190, activation_fn=tf.nn.sigmoid)


x_input = tf.placeholder(tf.float32, [None, 64, 64, 3])
y_input = tf.placeholder(tf.int32, [None, 1])
logits = predict(x_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=logits))
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_input, tf.argmax(logits, -1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(100001):
        batch_x, batch_y = get_next_batch()
        feed_dict = {x_input: batch_x, y_input: batch_y}
        sess.run(train_op, feed_dict=feed_dict)
        if step % 100 == 0:
            loss_val, accuracy_val = sess.run([loss, accuracy], feed_dict=feed_dict)
            print("step = {} , loss = {:.5f} , accuracy = {:.5f} ".format(step, loss_val, accuracy_val))
