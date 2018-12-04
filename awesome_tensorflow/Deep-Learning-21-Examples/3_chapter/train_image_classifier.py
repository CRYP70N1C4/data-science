import tensorflow as tf
from tensorflow.contrib import slim

dataset = None
network_fn = None
image_preprocessing_fn = None

provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[images, labels] = provider.get(['image', 'label'])
logits, end_points = network_fn(images)

slim.losses.softmax_cross_entropy(logits, labels, weights=1.0)
