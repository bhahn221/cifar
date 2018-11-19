import tensorflow as tf
import numpy as np
import argparse
import os
import time

import cifar_dataset
import utils

slim = tf.contrib.slim

MODEL='resnet20'

SKIP_STEP=100
CIFAR_TRAIN_SIZE=cifar_dataset.train_dataset_size
CIFAR_TEST_SIZE=cifar_dataset.test_dataset_size
TRAIN_BATCH_SIZE=128
TEST_BATCH_SIZE=100
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
N_EPOCH=1

def main(args):
    # define session
    sess = tf.Session()
    
    filename = '.'.join([tf.train.latest_checkpoint('checkpoints/'+MODEL), "meta"])
    saver = tf.train.import_meta_graph(filename)

    # load from checkpoint if it exists
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'+MODEL+'/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # get dropout & global_step
    graph = tf.get_default_graph()
    #dropout1 = graph.get_tensor_by_name('model/dropout1:0')
    #dropout2 = graph.get_tensor_by_name('model/dropout2:0')
    global_step = graph.get_tensor_by_name('train/global_step:0')
    batch_size = graph.get_tensor_by_name('input/batch_size:0')
    train_dataset_init = graph.get_operation_by_name('input/train_dataset_init')
    test_dataset_init = graph.get_operation_by_name('input/test_dataset_init')
    test_op = graph.get_tensor_by_name('test/Sum:0')
    learning_rate = graph.get_tensor_by_name('train/learning_rate:0')
    is_training = graph.get_tensor_by_name('resnet20/is_training:0')
    weight_decay = graph.get_tensor_by_name('loss/weight_decay:0')
    
    # prepair for testing
    start_time = time.time()
    n_batch = int(CIFAR_TEST_SIZE / TEST_BATCH_SIZE)

    sess.run(test_dataset_init, feed_dict={batch_size: TEST_BATCH_SIZE}) 
    
    # run test
    total_correct = 0
    for index in range(n_batch):
        batch_correct = sess.run(test_op,
                                 feed_dict={is_training: False,
                                            weight_decay: 0.0})
        
        total_correct += batch_correct

    print('Accuracy: {0}'.format(float(total_correct)/CIFAR_TEST_SIZE))
    print('Total time: {0} seconds'.format(time.time() - start_time))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test model on CIFAR')
    args = parser.parse_args()
    main(args)
