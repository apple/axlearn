"""Tensorflow data utils."""

import numpy as np
import tensorflow as tf


def get_dataset_cardinality(ds):
    batch_size = 1
    cardinality = np.sum([1 for i in ds.batch(batch_size)])
    return cardinality


# The following are copied from:
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
