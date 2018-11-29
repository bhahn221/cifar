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
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1

def main(args):
    # define session
    sess = tf.Session()
    
    # get input data
    x, y = cifar_dataset.inputs()
    graph = tf.get_default_graph()
    batch_size = graph.get_tensor_by_name('input/batch_size:0')
    train_dataset_init = graph.get_operation_by_name('input/train_dataset_init')
    test_dataset_init = graph.get_operation_by_name('input/test_dataset_init')
    
    # prepair for training dataset
    sess.run(train_dataset_init, feed_dict={batch_size: TRAIN_BATCH_SIZE}) 
    
    train_sum = np.zeros([32, 32, 3]).astype('int64')
    test_sum = np.zeros([32, 32, 3]).astype('int64')
    
    train_dev_sum = np.zeros([32, 32, 3]).astype('float32')
    test_dev_sum = np.zeros([32, 32, 3]).astype('float32')

    ####################################################################
    
    # run train dataset
    for index in range(CIFAR_TRAIN_SIZE):
        train_sum += np.squeeze(sess.run(x))

    print 'average of train image'
    train_mean = train_sum.astype('float32') / CIFAR_TRAIN_SIZE
    print train_mean

    # run train dataset
    for index in range(CIFAR_TRAIN_SIZE):
        train_dev_sum += ((np.squeeze(sess.run(x)) - train_mean) ** 2)

    print 'std-dev of train image'
    train_dev = np.sqrt(train_dev_sum.astype('float32') / CIFAR_TRAIN_SIZE)
    print train_dev

    ####################################################################

    ## prepair for testing dataset
    #sess.run(test_dataset_init, feed_dict={batch_size: TEST_BATCH_SIZE}) 
    #
    ## run test dataset
    #for index in range(CIFAR_TEST_SIZE):
    #    test_sum += np.squeeze(sess.run(x))

    #print 'average of test image'
    #test_mean = test_sum.astype('float32') / CIFAR_TEST_SIZE
    #
    ## run test dataset
    #for index in range(CIFAR_TEST_SIZE):
    #    test_dev_sum += ((np.squeeze(sess.run(x)) - test_mean) ** 2)

    #print 'std-dev of test image'
    #test_dev = np.sqrt(test_dev_sum.astype('float32')) / CIFAR_TEST_SIZE

    ####################################################################
    
    cifar_mean_image = {}
    cifar_mean_image['train_mean'] = train_mean
    cifar_mean_image['train_dev'] = train_dev
    #cifar_mean_image['test_mean'] = test_mean
    #cifar_mean_image['test_dev'] = test_dev

    f=open('cifar_mean_image.pickle', 'wb')
    pickle.dump(cifar_mean_image, f)
    f.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get CIFAR mean image')
    args = parser.parse_args()
    main(args)
