import tensorflow as tf
import numpy as np
import argparse
import os
import time

import cifar_dataset
import utils

slim = tf.contrib.slim

import resnet
MODEL='resnet18'

SKIP_STEP=100
CIFAR_TRAIN_SIZE=50000
CIFAR_TEST_SIZE=10000
BATCH_SIZE=128
LEARNING_RATE=1e-3
N_EPOCH=80

def main(args):
    # define session
    sess = tf.Session()
    
    # get input data
    x, y = cifar_dataset.inputs(BATCH_SIZE)

    # TODO define model
    logits = resnet.resnet18(x)

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
    train_dataset_init = graph.get_operation_by_name('input/train_dataset_init')
    test_dataset_init = graph.get_operation_by_name('input/test_dataset_init')
    learning_rate = graph.get_tensor_by_name('train/learning_rate:0')

    # prepair for training
    initial_step = sess.run(global_step)

    start_time = time.time()
    n_batch = int(CIFAR_TRAIN_SIZE / BATCH_SIZE)
    
    sess.run(train_dataset_init) 

    # run train
    total_loss = 0.0
    for index in range(initial_step, n_batch * N_EPOCH):
        if index % n_batch == 0:
            print('Epoch: {}'.format(index / n_batch))

        _, batch_summary, batch_loss = sess.run([train_op, summary_op, loss], 
                                                feed_dict={learning_rate: LEARNING_RATE})
        
        # write summary
        writer.add_summary(batch_summary, global_step=index)
        total_loss += batch_loss

        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.5f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/'+MODEL+'/'+MODEL, index)

    print('Optimization Finished!')
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # prepair for testing
    start_time = time.time()
    n_batch = int(CIFAR_TEST_SIZE / BATCH_SIZE)

    sess.run(test_dataset_init) 
    
    # run test
    total_correct = 0
    for index in range(n_batch):
        batch_summary, batch_correct = sess.run([summary_op, test_op])
        
        total_correct += batch_correct

    print('Accuracy: {0}'.format(float(total_correct)/CIFAR_TEST_SIZE))
    print('Total time: {0} seconds'.format(time.time() - start_time))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model on CIFAR')
    args = parser.parse_args()
    main(args)
