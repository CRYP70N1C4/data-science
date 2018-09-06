import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
a = trunc_normal(0.16)


def inception_v3_base(inputs, scope=None):
    end_points = {}
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
            net = slim.conv2d(inputs, 32, [3, 3], stride=2)
            net = slim.conv2d(net, 32, [3, 3])
            net = slim.conv2d(net, 64, [3, 3], padding='SAME')
            net = slim.max_pool2d(net, [3, 3], stride=2)

            net = slim.conv2d(net, 80, [1, 1])
            net = slim.conv2d(net, 192, [3, 3])
            net = slim.max_pool2d(net, [3, 3], stride=2)

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Mixed_5b'):
                branch_0 = slim.conv2d(net, 64, [1, 1])
                branch_1 = slim.conv2d(net, 48, [1, 1])
                branch_1 = slim.conv2d(branch_1, 64, [5, 5])
                branch_2 = slim.conv2d(net, 64, [1, 1])
                branch_2 = slim.conv2d(branch_2, 96, [3, 3])
                branch_2 = slim.conv2d(branch_2, 96, [3, 3])
                branch_3 = slim.avg_pool2d(net, [3, 3])
                branch_3 = slim.conv2d(branch_3, 32, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3])

            with tf.variable_scope('Mixed_5c'):
                branch_0 = slim.conv2d(net, 64, [1, 1])
                branch_1 = slim.conv2d(net, 48, [1, 1])
                branch_1 = slim.conv2d(branch_1, 64, [5, 5])
                branch_2 = slim.conv2d(net, 64, [1, 1])
                branch_2 = slim.conv2d(branch_2, 96, [3, 3])
                branch_2 = slim.conv2d(branch_2, 96, [3, 3])
                branch_3 = slim.avg_pool2d(net, [3, 3])
                branch_3 = slim.conv2d(branch_3, 64, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3])

            with tf.variable_scope('Mixed_5d'):
                branch_0 = slim.conv2d(net, 64, [1, 1])
                branch_1 = slim.conv2d(net, 48, [1, 1])
                branch_1 = slim.conv2d(branch_1, 64, [5, 5])
                branch_2 = slim.conv2d(net, 64, [1, 1])
                branch_2 = slim.conv2d(branch_2, 96, [3, 3])
                branch_2 = slim.conv2d(branch_2, 96, [3, 3])
                branch_3 = slim.avg_pool2d(net, [3, 3])
                branch_3 = slim.conv2d(branch_3, 64, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3])

            with tf.variable_scope('Mixed_6a'):
                branch_0 = slim.conv2d(net, 384, [3, 3], 2, padding='VALID')
                branch_1 = slim.conv2d(net, 64, [1, 1])
                branch_1 = slim.conv2d(branch_1, 96, [3, 3])
                branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2)
                branch_2 = slim.max_pool2d(net, [3, 3], 2, stride='VALID')

            net = tf.concat([branch_0, branch_1, branch_2], 3)

            with tf.variable_scope('Mixed_6b'):
                branch_0 = slim.conv2d(net, 192, [1, 1])
                branch_1 = slim.conv2d(net, 128, [1, 1])
                branch_1 = slim.conv2d(branch_1, 128, [1, 7])
                branch_1 = slim.conv2d(branch_1, 192, [7, 1])

                branch_2 = slim.conv2d(net, 128, [1, 1])
                branch_2 = slim.conv2d(branch_2, 128, [7, 1])
                branch_2 = slim.conv2d(branch_2, 128, [1, 7])
                branch_2 = slim.conv2d(branch_2, 128, [7, 1])
                branch_2 = slim.conv2d(branch_2, 192, [1, 7])

                branch_3 = slim.avg_pool2d(net, [3, 3])
                branch_3 = slim.conv2d(branch_3, 192, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_6c'):
                branch_0 = slim.conv2d(net, 192, [1, 1])
                branch_1 = slim.conv2d(net, 160, [1, 1])
                branch_1 = slim.conv2d(branch_1, 160, [1, 7])
                branch_1 = slim.conv2d(branch_1, 192, [7, 1])

                branch_2 = slim.conv2d(net, 160, [1, 1])
                branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                branch_2 = slim.conv2d(branch_2, 160, [1, 7])
                branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                branch_2 = slim.conv2d(branch_2, 192, [1, 7])

                branch_3 = slim.avg_pool2d(net, [3, 3])
                branch_3 = slim.conv2d(branch_3, 192, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_6d'):
                branch_0 = slim.conv2d(net, 192, [1, 1])
                branch_1 = slim.conv2d(net, 160, [1, 1])
                branch_1 = slim.conv2d(branch_1, 160, [1, 7])
                branch_1 = slim.conv2d(branch_1, 192, [7, 1])

                branch_2 = slim.conv2d(net, 160, [1, 1])
                branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                branch_2 = slim.conv2d(branch_2, 160, [1, 7])
                branch_2 = slim.conv2d(branch_2, 160, [7, 1])
                branch_2 = slim.conv2d(branch_2, 192, [1, 7])

                branch_3 = slim.avg_pool2d(net, [3, 3])
                branch_3 = slim.conv2d(branch_3, 192, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_6e'):
                branch_0 = slim.conv2d(net, 192, [1, 1])
                branch_1 = slim.conv2d(net, 192, [1, 1])
                branch_1 = slim.conv2d(branch_1, 192, [1, 7])
                branch_1 = slim.conv2d(branch_1, 192, [7, 1])

                branch_2 = slim.conv2d(net, 192, [1, 1])
                branch_2 = slim.conv2d(branch_2, 192, [7, 1])
                branch_2 = slim.conv2d(branch_2, 192, [1, 7])
                branch_2 = slim.conv2d(branch_2, 192, [7, 1])
                branch_2 = slim.conv2d(branch_2, 192, [1, 7])

                branch_3 = slim.avg_pool2d(net, [3, 3])
                branch_3 = slim.conv2d(branch_3, 192, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points['Mixed_6e'] = net

            with tf.variable_scope('Mixed_7a'):
                branch_0 = slim.conv2d(net, 192, [1, 1])
                branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2)

                branch_1 = slim.conv2d(net, 192, [1, 1])
                branch_1 = slim.conv2d(branch_1, 192, [1, 7])
                branch_1 = slim.conv2d(branch_1, 192, [7, 1])
                branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID')

                branch_2 = slim.max_pool2d(net, [3, 3], padding='VALID')

            net = tf.concat([branch_0, branch_1, branch_2], 3)

            with tf.variable_scope('Mixed_7b'):
                branch_0 = slim.conv2d(net, 320, [1, 1])
                branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2)

                branch_1 = slim.conv2d(net, 384, [1, 1])
                branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3]),
                                      slim.conv2d(branch_1, 384, [3, 1]), 3])

                branch_2 = slim.conv2d(net, 448, [1, 1])
                branch_2 = slim.conv2d(branch_2, 384, [3, 3])
                branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3]),
                                      slim.conv2d(branch_2, 384, [3, 1])], 3)

                branch_3 = slim.avg_pool2d(net, [3, 3])
                branch_3 = slim.conv2d(branch_3, 192, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            with tf.variable_scope('Mixed_7c'):
                branch_0 = slim.conv2d(net, 320, [1, 1])
                branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2)

                branch_1 = slim.conv2d(net, 384, [1, 1])
                branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3]),
                                      slim.conv2d(branch_1, 384, [3, 1]), 3])

                branch_2 = slim.conv2d(net, 448, [1, 1])
                branch_2 = slim.conv2d(branch_2, 384, [3, 3])
                branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3]),
                                      slim.conv2d(branch_2, 384, [3, 1])], 3)

                branch_3 = slim.avg_pool2d(net, [3, 3])
                branch_3 = slim.conv2d(branch_3, 192, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        return net, end_points;


def inception_v3(inputs, num_classes=1000, is_training=True, scope='InceptionV3',
                 reuse=None, spatial_squeeze=None,
                 dropout_keep_prob=0.8
                 , prediction_fn=slim.softmax):
    with tf.variable_scope(scope, [inputs, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v3_base(inputs, scope=scope)
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID')
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1])
                    aux_logits = slim.conv2d(aux_logits, 768, [5, 5], padding='VALID')
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None)
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2])
                    end_points['AuxLogits'] = aux_logits

                with tf.variable_scope('Logits'):
                    net = slim.avg_pool2d(net, [8, 8], padding='VALID')
                    net = slim.dropout(net, keep_prob=dropout_keep_prob)
                    end_points['PreLogits'] = net
                    logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None)

                    if spatial_squeeze:
                        logits = tf.squeeze(logits, [1, 2])

                    end_points['Logits'] = logits
                    end_points['Predictions'] = prediction_fn(logits, scope='Prediction')

                return logits, end_points
