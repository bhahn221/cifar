# parse_example is faster: https://github.com/tensorflow/tensorflow/pull/14751
# code I used to make TFRecords for ImageNet: https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py
# 7x with Dataset API: https://sebastianwallkoetter.wordpress.com/2018/02/24/optimize-tf-input-pipeline/
# AlexNet paper to refer for data augmentation: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

import tensorflow as tf
import numpy as np
import pickle

import matplotlib.pyplot as plt # testing functionality of pipeline

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

def normalize(image, label):
    normalized_image = (tf.cast(image, tf.float32) - cifar_mean_image['train_mean']) / cifar_mean_image['train_dev']

    return normalized_image, label

def random_flip(image, label):
    flipped_image = tf.image.random_flip_left_right(image)

    return flipped_image, label

def change_light(image, label):
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

    return image, label

def random_crop(image, label):
    padded_image = tf.pad(image, [[2,2], [2,2], [0,0]], 'CONSTANT')
    cropped_image = tf.random_crop(padded_image, [32, 32, 3])

    return cropped_image, label

def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        }
    )

    image = tf.cast(image_ops.decode_image(features['image/encoded'], channels=3), tf.uint8)
    image = array_ops.reshape(image, [32, 32, 3])
    label = tf.cast(features['image/class/label'], tf.int32)

    return image, label

def inputs():
    with tf.name_scope('input'):
        batch_size = tf.placeholder(tf.int64, name='batch_size')

        train_dataset = tf.data.TFRecordDataset(MNIST_DIRECTORY+TRAIN_FILE)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.map(decode)
        train_dataset = train_dataset.map(random_flip)
        train_dataset = train_dataset.map(random_crop)
        train_dataset = train_dataset.map(normalize)
        train_dataset = train_dataset.shuffle(1000+batch_size*3)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(batch_size*5)

        test_dataset = tf.data.TFRecordDataset(MNIST_DIRECTORY+TEST_FILE)
        test_dataset = test_dataset.map(decode)
        test_dataset = test_dataset.map(normalize)
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.prefetch(batch_size*5)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_dataset_init_op = iterator.make_initializer(train_dataset, name='train_dataset_init')
        test_dataset_init_op = iterator.make_initializer(test_dataset, name='test_dataset_init')
    return iterator.get_next()

# Test code for the dataset
def main():
    sess = tf.Session()
    
    x, y = inputs()
    
    graph = tf.get_default_graph()
    batch_size = graph.get_tensor_by_name('input/batch_size:0')
    train_dataset_init = graph.get_operation_by_name('input/train_dataset_init')
    test_dataset_init = graph.get_operation_by_name('input/test_dataset_init')
    
    sess.run(train_dataset_init, feed_dict={batch_size: 2}) 
    #sess.run(val_dataset_init, feed_dict={batch_size: 2}) 
    
    image, label = sess.run([x, y])
    print label[0], get_human_readable_label(label[0])
    print np.max(image[0])
    plt.imshow(image[0])
    plt.show()
    print label[1], get_human_readable_label(label[1])
    print np.max(image[1])
    plt.imshow(image[1])
    plt.show()

if __name__ == '__main__':
    main()
