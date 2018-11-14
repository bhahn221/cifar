import tensorflow as tf
import numpy as np
import argparse
import os
import time

import cifar_dataset
import utils

slim = tf.contrib.slim

import model
MODEL='model'

SKIP_STEP=100
CIFAR_TRAIN_SIZE=50000
CIFAR_TEST_SIZE=10000
BATCH_SIZE=100
LEARNING_RATE=1e-3
N_EPOCH=1
PRUNING_ITERATION=40
PRUNING_RATE=70

def get_kernels(graph=tf.get_default_graph()):
    vals = [val for op in graph.get_operations() for val in op.values()]
    return filter(lambda val: 'kernel:0' in val.name, vals)

def prune(weight, pruning_rate):
    pruned_weight = np.copy(weight)
    threshold = np.percentile(np.abs(weight), pruning_rate)
    pruned_weight[np.abs(weight) < threshold] = 0

    # scale the weight so that the sum would be the same
    # NOTE explodes
    #pruned_weight = 1. / (1 - float(PRUNING_RATE)/100) * pruned_weight
    
    return pruned_weight

def main(args):
    # define session
    sess = tf.Session()
    
    filename = '.'.join([tf.train.latest_checkpoint('checkpoints/'+MODEL), "meta"])
    saver = tf.train.import_meta_graph(filename)

    # save summary
    writer = tf.summary.FileWriter('./graphs/'+MODEL, sess.graph)
    
    # load from checkpoint if it exists
    if os.path.exists('checkpoints/'+MODEL+'_pruned'):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'+MODEL+'_pruned/checkpoint'))
    else:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'+MODEL+'/checkpoint'))
        utils.make_dir('checkpoints/'+MODEL+'_pruned')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # get dropout & global_step
    graph = tf.get_default_graph()
    dropout1 = graph.get_tensor_by_name('model/dropout1:0')
    dropout2 = graph.get_tensor_by_name('model/dropout2:0')
    global_step = graph.get_tensor_by_name('train/global_step:0')
    train_dataset_init = graph.get_operation_by_name('input/train_dataset_init')
    test_dataset_init = graph.get_operation_by_name('input/test_dataset_init')
    test_op = graph.get_tensor_by_name('test/Sum:0')
    
    loss = graph.get_tensor_by_name('loss/loss:0')
    train_op = graph.get_tensor_by_name('train/train_op:0')
    summary_op = graph.get_tensor_by_name('summary/Merge/MergeSummary:0')
    learning_rate = graph.get_tensor_by_name('train/learning_rate:0')

    for iteration in range(PRUNING_ITERATION):
        """ STEP 1 : PRUNE """
        # save kernels into dict
        kernels_dict = {}
        for kernel in get_kernels():
            kernels_dict[kernel.name] = sess.run(kernel)

        # save pruned kernels (array) into dict
        pruned_kernels_dict = {}
        for name, array in kernels_dict.iteritems():
            pruned_kernels_dict[name] = prune(array, PRUNING_RATE)

        # assign pruned kernels (array) to graph
        for name, array in pruned_kernels_dict.iteritems():
            print 'pruned {}'.format(name)
            kernel = graph.get_tensor_by_name(name)
            sess.run(tf.assign(kernel, array))

        print('Pruning Finished!')

        """ STEP 2 : RE-TRAIN """
        # prepair for training
        initial_step = sess.run(global_step)

        start_time = time.time()
        n_batch = int(CIFAR_TRAIN_SIZE / BATCH_SIZE)
        
        sess.run(train_dataset_init) 

        # run train
        total_loss = 0.0
        for index in range(initial_step, initial_step + n_batch * N_EPOCH):
            if index % n_batch == 0:
                # assume it has been trained 120 epochs before pruning
                print('Epoch: {}'.format((index - 60000) / n_batch))

            _, batch_summary, batch_loss = sess.run([train_op, summary_op, loss], 
                                                    feed_dict={dropout1: 1.0,
                                                               dropout2: 1.0,
                                                               learning_rate: LEARNING_RATE})
            
            # write summary
            writer.add_summary(batch_summary, global_step=index)
            total_loss += batch_loss

            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.5f}'.format(index + 1, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, 'checkpoints/'+MODEL+'/pruned_'+MODEL, index)

        print('Optimization Finished!')
        print('Total time: {0} seconds'.format(time.time() - start_time))

        """ STEP 3 : PRUNE """
        # save kernels into dict
        kernels_dict = {}
        for kernel in get_kernels():
            kernels_dict[kernel.name] = sess.run(kernel)

        # save pruned kernels (array) into dict
        pruned_kernels_dict = {}
        for name, array in kernels_dict.iteritems():
            pruned_kernels_dict[name] = prune(array, PRUNING_RATE)

        # assign pruned kernels (array) to graph
        for name, array in pruned_kernels_dict.iteritems():
            print 'pruned {}'.format(name)
            kernel = graph.get_tensor_by_name(name)
            sess.run(tf.assign(kernel, array))

        print('Pruning Finished!')

        """ STEP 4 : CHECK ACCURACY """
        # prepair for testing
        start_time = time.time()
        n_batch = int(CIFAR_TEST_SIZE / BATCH_SIZE)

        sess.run(test_dataset_init) 
        
        # run test
        total_correct = 0
        for index in range(n_batch):
            batch_summary, batch_correct = sess.run([summary_op, test_op],
                                                 feed_dict={dropout1: 1.0,
                                                            dropout2: 1.0})
            
            total_correct += batch_correct

        print('Accuracy: {0}'.format(float(total_correct)/CIFAR_TEST_SIZE))
        print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Iterative Pruning Finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model on CIFAR')
    args = parser.parse_args()
    main(args)
