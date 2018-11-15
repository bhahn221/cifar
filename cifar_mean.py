import tensorflow as tf
import numpy as np
import argparse
import os
import time
import pickle

import cifar_dataset
import utils

CIFAR_TRAIN_SIZE=50000
CIFAR_TEST_SIZE=10000

def main(args):
    # define session
    sess = tf.Session()
    
    # get input data
    x, y = cifar_dataset.inputs(1)
    graph = tf.get_default_graph()
    train_dataset_init = graph.get_operation_by_name('input/train_dataset_init')
    test_dataset_init = graph.get_operation_by_name('input/test_dataset_init')
    
    # prepair for training dataset
    sess.run(train_dataset_init) 
    
    train_sum = np.zeros([32, 32, 3]).astype('int64')
    test_sum = np.zeros([32, 32, 3]).astype('int64')

    # run train dataset
    for index in range(CIFAR_TRAIN_SIZE):
        train_sum += np.squeeze(sess.run(x))

    print 'average of train image'
    train_mean = train_sum.astype('float32') / CIFAR_TRAIN_SIZE
    print train_mean

    # prepair for testing dataset
    sess.run(test_dataset_init) 
    
    # run test dataset
    for index in range(CIFAR_TEST_SIZE):
        test_sum += np.squeeze(sess.run(x))

    print 'average of test image'
    test_mean = test_sum.astype('float32') / CIFAR_TEST_SIZE
    print test_mean

    cifar_mean_image = {}
    cifar_mean_image['train'] = train_mean
    cifar_mean_image['test'] = test_mean

    f=open('cifar_mean_image.pickle', 'wb')
    pickle.dump(cifar_mean_image, f)
    f.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get CIFAR mean image')
    args = parser.parse_args()
    main(args)
