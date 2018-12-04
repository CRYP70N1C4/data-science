import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A named tuple describing cifar_10.py ResNet block'


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


        # def conv2d
