# parse_example is faster: https://github.com/tensorflow/tensorflow/pull/14751
# code I used to make TFRecords for ImageNet: https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py
# 7x with Dataset API: https://sebastianwallkoetter.wordpress.com/2018/02/24/optimize-tf-input-pipeline/
# AlexNet paper to refer for data augmentation: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

import tensorflow as tf
import numpy as np
import pickle

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops

MNIST_DIRECTORY = '/home/bhahn221/dataset/cifar10/'
TRAIN_FILE      = 'cifar10_train.tfrecord'
TEST_FILE       = 'cifar10_test.tfrecord'

train_dataset_size = 50000
test_dataset_size = 10000

f = open('cifar_mean_image.pickle', 'rb')
cifar_mean_image = pickle.load(f)
f.close()

def normalize_train(image, label):
    image = (tf.cast(image, tf.float32) - cifar_mean_image['train']) / 255

    return image, label

def normalize_test(image, label):
    image = (tf.cast(image, tf.float32) - cifar_mean_image['test']) / 255

    return image, label

def augment(image, label):
    flipped_image = tf.image.random_flip_left_right(image)
    padded_image = tf.pad(flipped_image, [[0,0], [4,4], [4,4], [0,0]], 'CONSTANT')
    cropped_image = tf.random_crop(padded_image, [1, 32, 32, 3])

    return image, label

def decode(serialized_example):
    features = tf.parse_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        }
    )

    image = tf.map_fn(lambda x: image_ops.decode_image(x, channels=3), features['image/encoded'], dtype=tf.uint8)
    image = tf.map_fn(lambda x: array_ops.reshape(x, [32, 32, 3]), image)
    label = tf.cast(features['image/class/label'], tf.int32)

    return image, label

def inputs(batch_size):
    with tf.name_scope('input'):
        train_dataset = tf.data.TFRecordDataset(MNIST_DIRECTORY+TRAIN_FILE)
        train_dataset = train_dataset.repeat(None)
        train_dataset = train_dataset.shuffle(50000)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(decode)
        train_dataset = train_dataset.map(normalize_train) # commented while calculating mean
        train_dataset = train_dataset.map(augment)
        train_dataset = train_dataset.prefetch(batch_size*5)

        test_dataset = tf.data.TFRecordDataset(MNIST_DIRECTORY+TEST_FILE)
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.map(decode)
        test_dataset = test_dataset.map(normalize_test) # commented while calculating mean
        test_dataset = test_dataset.prefetch(batch_size*5)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_dataset_init_op = iterator.make_initializer(train_dataset, name='train_dataset_init')
        test_dataset_init_op = iterator.make_initializer(test_dataset, name='test_dataset_init')
    return iterator.get_next()
