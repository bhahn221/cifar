import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def _fixed_padding(x, kernel):
  pad_total = kernel - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  y = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  return y

def _conv2d(x, out_channel, kernel, stride, pad='SAME', bias=False, name=None):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        if stride > 1:
            x = _fixed_padding(x, kernel)

        with tf.device('/CPU:0'):
            kernel = tf.get_variable('kernel',
                                     [kernel, kernel, in_shape[3], out_channel],
                                     tf.float32,
                                     initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(2./kernel/kernel/out_channel)))
        if kernel not in tf.get_collection('WEIGHT_DECAY'):
            tf.add_to_collection('WEIGHT_DECAY', kernel)
        # for ResNet, this seems like the norm
        # https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
        conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], ('SAME' if stride == 1 else 'VALID'))
    
        if bias == True:
            b = tf.get_variable('bias', 
                                [out_channel],
                                tf.float32,
                                initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)
        
    return conv

def _max_pool(x, kernel, stride, pad='SAME', name='name'):
    pool = tf.nn.max_pool(x,
                          [1, kernel, kernel, 1],
                          [1, stride, stride, 1],
                          pad, name=name)
    return pool

def _avg_pool(x, kernel, stride, pad='SAME', name='name'):
    pool = tf.nn.avg_pool(x,
                          [1, kernel, kernel, 1],
                          [1, stride, stride, 1],
                          pad, name=name)
    return pool

def _bn(x, is_training, name=None):
    moving_average_decay = 0.9
    with tf.variable_scope(name):
        decay = moving_average_decay
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu',
                                 batch_mean.get_shape(),
                                 tf.float32,
                                 initializer=tf.zeros_initializer(),
                                 trainable=False)
            sigma = tf.get_variable('sigma',
                                    batch_var.get_shape(),
                                    tf.float32,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)
            beta = tf.get_variable('beta', 
                                 batch_mean.get_shape(),
                                 tf.float32,
                                 initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma',
                                    batch_var.get_shape(),
                                    tf.float32,
                                    initializer=tf.ones_initializer())
        update = 1.0 - decay
        mu = mu.assign_sub(update * (mu - batch_mean))
        sigma = sigma.assign_sub(update * (sigma - batch_var))

        mean, var = tf.cond(tf.cast(is_training, tf.bool), 
                                    lambda: (batch_mean, batch_var), 
                                    lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return bn

def _relu(x, leaky=0.0, name=None):
    if leaky > 0.0:
        return tf.maximum(x, x*leaky, name=name)
    else:
        return tf.nn.relu(x, name=name)

def _fc(x, out_shape, name='name'):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        with tf.device('/CPU:0'):
            w = tf.get_variable('weight', 
                                [in_shape[1], out_shape],
                                tf.float32,
                                initializer=tf.random_normal_initializer(
                                    stddev=np.sqrt(2./out_shape)))
            b = tf.get_variable('bias', 
                                [out_shape],
                                tf.float32,
                                initializer=tf.constant_initializer(0.0))
        if w not in tf.get_collection('WEIGHT_DECAY'):
            tf.add_to_collection('WEIGHT_DECAY', w)
        # regulariazation is usually not applied to the bias terms
        # https://stats.stackexchange.com/questions/153605/no-regularisation-term-for-bias-unit-in-neural-network
        #if b not in tf.get_collection('WEIGHT_DECAY'):
        #    tf.add_to_collection('WEIGHT_DECAY', b)
        fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc

def _resblk_first(x, out_channel, kernel, stride, is_training=True, name='unit'):
    in_channel = x.get_shape().as_list()[-1]
    projection = False
    with tf.variable_scope(name):
        if stride == 1:
            shortcut = tf.identity(x, name='shortcut')
        else:
            # apparently resnet do not use max pooling
            #shortcut = _max_pool(x, 3, stride, 'SAME', name='shortcut')
            shortcut = _avg_pool(x, 3, stride, 'SAME', name='shortcut')
        if in_channel != out_channel:
            if projection == True:
                with tf.variable_scope('shortcut'):
                    shortcut = _conv2d(shortcut, out_channel, 1, stride, 'SAME', name='conv')
                    shortcut = _bn(shortcut, is_training, name='bn')
            else:
                shortcut = tf.pad(shortcut, [[0,0], [0,0], [0,0], [0, out_channel - in_channel]], name='shortcut')

        x = _conv2d(x, out_channel, kernel, stride, 'SAME', name='conv1')
        x = _bn(x, is_training, name='bn1')
        x = _relu(x, 0.0, name='relu1')
        x = _conv2d(x, out_channel, kernel, 1, 'SAME', name='conv2')
        x = _bn(x, is_training, name='bn2')
        
        x = x + shortcut
        
        x = _relu(x, 0.0, name='relu2')

    return x

def _resblk(x, out_channel, kernel, is_training=True, name='unit'):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        shortcut = tf.identity(x)

        x = _conv2d(x, out_channel, kernel, 1, name='conv1')
        x = _bn(x, is_training, name='bn1')
        x = _relu(x, 0.0, name='relu1')
        x = _conv2d(x, out_channel, kernel, 1, name='conv2')
        x = _bn(x, is_training, name='bn2')
        
        x = x + shortcut
        
        x = _relu(x, 0.0, name='relu2')

    return x


def resnet20(x):
    with tf.variable_scope('resnet20'):
        is_training = tf.placeholder(tf.bool, [], 'is_training')

        with tf.variable_scope('conv1'):
            # bias is not used for the convolutions because batch norm layers
            # deal with both scaling and the shifting of the output.
            x = _conv2d(x, 16, 3, 1, name='conv1')
            x = _bn(x, is_training, name='bn1')
            x = _relu(x, 0.0, name='relu1')

        with tf.variable_scope('conv2'):
            x = _resblk(x, 16, 3, is_training=is_training, name='conv2_1')
            x = _resblk(x, 16, 3, is_training=is_training, name='conv2_2')
            x = _resblk(x, 16, 3, is_training=is_training, name='conv2_3')
        
        with tf.variable_scope('conv3'):
            x = _resblk_first(x, 32, 3, 2, is_training=is_training, name='conv3_1')
            x = _resblk(x, 32, 3, is_training=is_training, name='conv3_2')
            x = _resblk(x, 32, 3, is_training=is_training, name='conv3_3')

        with tf.variable_scope('conv4'):
            x = _resblk_first(x, 64, 3, 2, is_training=is_training, name='conv4_1')
            x = _resblk(x, 64, 3, is_training=is_training, name='conv4_2')
            x = _resblk(x, 64, 3, is_training=is_training, name='conv4_3')

        with tf.variable_scope('fc5'):
            x = _avg_pool(x, 8, 8, name='gap')
            x = tf.layers.flatten(x)
            logits = _fc(x, 10, name='fc5')
    return logits

def loss(logits, labels):
    with tf.name_scope('loss'):
        # softmax loss
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
        
        # weight decay
        weight_decay = tf.placeholder(tf.float32, name='weight_decay')
        
        costs = [tf.nn.l2_loss(var) for var in tf.get_collection('WEIGHT_DECAY')]
        l2_loss = tf.multiply(weight_decay, tf.add_n(costs))
        
        loss += l2_loss

        # optimizer
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        #optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer')
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
                                               momentum=0.9,
                                               use_nesterov=True,
                                               name='optimizer')
        train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')
    
    return train_op

def testing(logits, labels):
    with tf.name_scope('test'):
        correct = tf.nn.in_top_k(logits, labels, 1)
        test_op = tf.reduce_sum(tf.cast(correct, tf.int32))

    return test_op
