import tensorflow as tf
import numpy as np
import argparse
import os
import time

import cifar_dataset
import utils

slim = tf.contrib.slim

import resnet
MODEL='resnet20'

SKIP_STEP=100
CIFAR_TRAIN_SIZE=cifar_dataset.train_dataset_size
CIFAR_TEST_SIZE=cifar_dataset.test_dataset_size
TRAIN_BATCH_SIZE=128
TEST_BATCH_SIZE=500
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-3
N_EPOCH=164
LOG='train_log'

def log(file_name, message):
    print message
    with open(LOG, "a") as myfile:
        myfile.write(message+'\n')

def main(args):
    # define session
    sess = tf.Session()
    
    # get input data
    x, y = cifar_dataset.inputs()

    # TODO define model
    logits = resnet.resnet20(x)

    # TODO define loss
    loss = resnet.loss(logits, y)

    # TODO define summary
    summary_op = resnet.summary(loss)
    
    # TODO define training & testing
    train_op = resnet.training(loss)
    test_op = resnet.testing(logits, y)

    # initializer
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # save graph
    saver = tf.train.Saver()
    
    # save summary
    writer = tf.summary.FileWriter('./graphs/'+MODEL, sess.graph)

    # load from checkpoint if it exists
    utils.make_dir('checkpoints')
    utils.make_dir('checkpoints/'+MODEL)
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
    learning_rate = graph.get_tensor_by_name('train/learning_rate:0')
    is_training = graph.get_tensor_by_name('resnet20/is_training:0')
    weight_decay = graph.get_tensor_by_name('loss/weight_decay:0')

    # prepair for training
    initial_step = sess.run(global_step)

    start_time = time.time()
    n_batch = int(CIFAR_TRAIN_SIZE / TRAIN_BATCH_SIZE)
    
    sess.run(train_dataset_init, feed_dict={batch_size: TRAIN_BATCH_SIZE}) 

    # run train
    total_loss = 0.0
    for index in range(initial_step, n_batch * N_EPOCH):
        if index % n_batch == 0:
            #print('Epoch: {}'.format(index / n_batch))
            log(LOG, 'Epoch: {}'.format(index / n_batch))

        _, batch_summary, batch_loss = sess.run([train_op, summary_op, loss], 
                                                feed_dict={learning_rate: LEARNING_RATE,
                                                           is_training: True,
                                                           weight_decay: WEIGHT_DECAY})
        
        # write summary
        writer.add_summary(batch_summary, global_step=index)
        total_loss += batch_loss

        if (index + 1) % SKIP_STEP == 0:
            #print('Average loss at step {}: {:5.5f}'.format(index + 1, total_loss / SKIP_STEP))
            log(LOG, 'Average loss at step {}: {:5.5f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/'+MODEL+'/'+MODEL, index)

        if (index + 1) % (n_batch * 10) == 0:
            #print('Check Test Accuracy')
            log(LOG, 'Check Test Accuracy')

            # prepair for testing
            _start_time = time.time()
            _n_batch = int(CIFAR_TEST_SIZE / TEST_BATCH_SIZE)
            
            sess.run(test_dataset_init, feed_dict={batch_size: TEST_BATCH_SIZE}) 
            
            # run test
            total_correct = 0
            for index in range(_n_batch):
                batch_summary, batch_correct = sess.run([summary_op, test_op],
                                                        feed_dict={is_training: False,
                                                                   weight_decay: 0.0})
                
                total_correct += batch_correct

            #print('Accuracy: {0}'.format(float(total_correct)/CIFAR_TEST_SIZE))
            #print('Total time: {0} seconds'.format(time.time() - _start_time))
            log(LOG, 'Accuracy: {0}'.format(float(total_correct)/CIFAR_TEST_SIZE))
            log(LOG, 'Total time: {0} seconds'.format(time.time() - _start_time))

            # revert changes made to parameters
            sess.run(train_dataset_init, feed_dict={batch_size: TRAIN_BATCH_SIZE}) 

    #print('Optimization Finished!')
    #print('Total time: {0} seconds'.format(time.time() - start_time))
    log(LOG, 'Optimization Finished!')
    log(LOG, 'Total time: {0} seconds'.format(time.time() - start_time))

    # prepair for testing
    start_time = time.time()
    n_batch = int(CIFAR_TEST_SIZE / TEST_BATCH_SIZE)

    sess.run(test_dataset_init, feed_dict={batch_size: TEST_BATCH_SIZE}) 
    
    # run test
    total_correct = 0
    for index in range(n_batch):
        batch_summary, batch_correct = sess.run([summary_op, test_op],
                                                feed_dict={is_training: False,
                                                           weight_decay: 0.0})
        
        total_correct += batch_correct

    #print('Accuracy: {0}'.format(float(total_correct)/CIFAR_TEST_SIZE))
    #print('Total time: {0} seconds'.format(time.time() - start_time))
    log(LOG, 'Accuracy: {0}'.format(float(total_correct)/CIFAR_TEST_SIZE))
    log(LOG, 'Total time: {0} seconds'.format(time.time() - start_time))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model on CIFAR')
    args = parser.parse_args()
    main(args)
