import tensorflow as tf


def make_multi_cell(state_dim, num_layers):
    cells = [tf.contrib.rnn.LSTMCell(state_dim) for _ in range(num_layers)]
    return tf.contrib.rnn.MultiRNNCell(cells)


x_input = tf.placeholder(tf.float32, [None, 5, 3])
multi_cell = make_multi_cell(state_dim=10, num_layers=6)
outputs, states = tf.nn.dynamic_rnn(multi_cell, x_input, dtype=tf.float32)

print(x_input)
print(outputs)