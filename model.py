import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def model(x):
    with tf.variable_scope('model'):
        dropout1 = tf.placeholder(tf.float32, name='dropout1')
        dropout2 = tf.placeholder(tf.float32, name='dropout2')
        
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, name='conv1')
            conv1 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv2')
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='SAME', name='pool1')
            conv1 = tf.layers.dropout(conv1, rate=dropout1)

        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv2d(conv1, 128, 3, activation=tf.nn.relu, name='conv1')
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='SAME', name='pool1')
            conv2 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu, name='conv2')
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='SAME', name='pool2')
            conv2 = tf.layers.dropout(conv2, rate=dropout1)
        
        with tf.variable_scope('fc3'):
            fc1 = tf.contrib.layers.flatten(conv2)
            fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc')
            fc1 = tf.layers.dropout(fc1, rate=dropout2)

        with tf.variable_scope('fc4'):
            logits = tf.layers.dense(fc1, 10, name='fc4')

    return logits

def loss(logits, labels):
    with tf.name_scope('loss'):
        entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.reduce_mean(entropy, name='loss')

    return loss

def summary(loss):
    with tf.name_scope('summary'):
        tf.summary.scalar('loss', loss)
        tf.summary.histogram('histogram loss', loss)
        summary_op = tf.summary.merge_all()

    return summary_op

def training(loss):
    with tf.name_scope('train'):
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer')
        train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')
    
    return train_op

def testing(logits, labels):
    with tf.name_scope('test'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        test_op = tf.reduce_sum(tf.cast(correct, tf.int32))

    return test_op
