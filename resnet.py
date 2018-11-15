import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def _resblk_first(x, out_channel, kernel, stride, name='unit'):
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        if in_channel == out_channel:
            if stride == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.layers.max_pooling2d(x, stride, stride, padding='VALID')
        else:
            shortcut = tf.layers.conv2d(x, out_channel, 1, (stride, stride), name='shortcut')

        x = tf.layers.conv2d(x, out_channel, kernel, (stride, stride), padding='SAME', name='conv1')
        x = tf.layers.batch_normalization(x, name='bn1')
        x = tf.nn.relu(x, name='relu1')
        x = tf.layers.conv2d(x, out_channel, kernel, padding='SAME', name='conv2')
        x = tf.layers.batch_normalization(x, name='bn2')
        
        x = x + shortcut
        x = tf.nn.relu(x, name='relu2')
    
    return x

def _resblk(x, kernel, name='unit'):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        shortcut = tf.identity(x)
        
        x = tf.layers.conv2d(x, num_channel, kernel, padding='SAME', name='conv1')
        x = tf.layers.batch_normalization(x, name='bn1')
        x = tf.nn.relu(x, name='relu1')
        x = tf.layers.conv2d(x, num_channel, kernel, padding='SAME', name='conv2')
        x = tf.layers.batch_normalization(x, name='bn2')

        x = x + shortcut
        x = tf.nn.relu(x, name='relu2')

    return x

def resnet18(x):
    with tf.variable_scope('resnet18'):
        with tf.variable_scope('conv1'):
            #x = tf.layers.conv2d(x, 64, 7, (2, 2), name='conv1') # for imagenet
            x = tf.layers.conv2d(x, 64, 3, (1, 1), name='conv1') # for cifar10
            x = tf.layers.batch_normalization(x, name='bn1')
            #x = tf.nn.relu(x, name='relu1') # for imagenet
            #x = tf.layers.max_pooling2d(x, 3, 2, padding='SAME', name='pool1') # for imagenet

        with tf.variable_scope('conv2'):
            x = _resblk(x, 3, name='conv2_1')
            x = _resblk(x, 3, name='conv2_2')
        
        with tf.variable_scope('conv3'):
            x = _resblk_first(x, 128, 3, 2, name='conv3_1')
            x = _resblk(x, 3, name='conv3_2')

        with tf.variable_scope('conv4'):
            x = _resblk_first(x, 256, 3, 2, name='conv4_1')
            x = _resblk(x, 3, name='conv4_2')

        with tf.variable_scope('conv5'):
            x = _resblk_first(x, 512, 3, 2, name='conv5_1')
            x = _resblk(x, 3, name='conv5_2')

        with tf.variable_scope('fc6'):
            x = tf.reduce_mean(x, axis=[1,2])
            logits = tf.layers.dense(x, 10, name='fc6')

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
