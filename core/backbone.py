#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-02-17 11:03:35
#   Description :
#
#================================================================

import core.common as common
import tensorflow as tf

import os, scipy.io
import numpy as np

import opt

class vgg16:
    def __init__(self, reuse=False):
        self.parameters = []
        self.reuse = reuse
        self.set_parameters()

    def set_parameters(self):
        with tf.variable_scope('vgg_conv1_1', reuse=self.reuse ):
            self.kernel11 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases11 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')

            self.parameters += [self.kernel11, self.biases11]

        with tf.variable_scope('vgg_conv1_2', reuse=self.reuse ):
            self.kernel12 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases12 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel12, self.biases12]

        # conv2_1
        with tf.variable_scope('vgg_conv2_1', reuse=self.reuse ):
            self.kernel21 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases21 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel21, self.biases21]

        # conv2_2
        with tf.variable_scope('vgg_conv2_2', reuse=self.reuse ):
            self.kernel22 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases22 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel22, self.biases22]

        # conv3_1
        with tf.variable_scope('vgg_conv3_1', reuse=self.reuse ):
            self.kernel31 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases31 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel31, self.biases31]

        # conv3_2
        with tf.variable_scope('vgg_conv3_2', reuse=self.reuse):
            self.kernel32 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases32 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel32, self.biases32]

        # conv3_3
        with tf.variable_scope('vgg_conv3_3', reuse=self.reuse ):
            self.kernel33 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases33 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel33, self.biases33]

        # conv4_1
        with tf.variable_scope('vgg_conv4_1', reuse=self.reuse ):
            self.kernel41 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases41 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel41, self.biases41]

        # conv4_2
        with tf.variable_scope('vgg_conv4_2', reuse=self.reuse ):
            self.kernel42 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases42 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel42, self.biases42]

        # conv4_3
        with tf.variable_scope('vgg_conv4_3', reuse=self.reuse ):
            self.kernel43 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases43 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel43, self.biases43]

        # conv5_1
        with tf.variable_scope('vgg_conv5_1', reuse=self.reuse ):
            self.kernel51 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases51 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel51, self.biases51]

        # conv5_2
        with tf.variable_scope('vgg_conv5_2', reuse=self.reuse):
            self.kernel52 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases52 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel52, self.biases52]

        # conv5_3
        with tf.variable_scope('vgg_conv5_3', reuse=self.reuse):
            self.kernel53 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            self.biases53 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [self.kernel53, self.biases53]

    def forward(self, imgs, trainable, reuse=False):

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = imgs-mean

        # conv1_1
        conv = tf.nn.conv2d(images, self.kernel11, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases11)
        out = opt.batch_norm(out, name='vgg_bn11', training=trainable)
        self.conv1_1 = tf.nn.relu(out)

        # conv1_2
        conv = tf.nn.conv2d(self.conv1_1, self.kernel12, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases12)
        out = opt.batch_norm(out, name='vgg_bn12', training=trainable)
        self.conv1_2 = tf.nn.relu(out)

        self.layer1 = attention_layer(self.conv1_1, self.conv1_2, None, bpool=False, reuse=reuse, name='od_att1')

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        conv = tf.nn.conv2d(self.pool1, self.kernel21, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases21)
        out = opt.batch_norm(out, name='vgg_bn21', training=trainable)
        self.conv2_1 = tf.nn.relu(out)

        # conv2_2
        conv = tf.nn.conv2d(self.conv2_1, self.kernel22, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases22)
        out = opt.batch_norm(out, name='vgg_bn22', training=trainable)
        self.conv2_2 = tf.nn.relu(out)

        self.layer2 = attention_layer(self.conv2_1, self.conv2_2, self.layer1, bpool=True, reuse=reuse, name='od_att2')

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        conv = tf.nn.conv2d(self.pool2, self.kernel31, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases31)
        out = opt.batch_norm(out, name='vgg_bn31', training=trainable)
        self.conv3_1 = tf.nn.relu(out)

        conv = tf.nn.conv2d(self.conv3_1, self.kernel32, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases32)
        out = opt.batch_norm(out, name='vgg_bn32', training=trainable)
        self.conv3_2 = tf.nn.relu(out)

        # conv3_3
        conv = tf.nn.conv2d(self.conv3_2, self.kernel33, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases33)
        out = opt.batch_norm(out, name='vgg_bn33', training=trainable)
        self.conv3_3 = tf.nn.relu(out)

        self.layer3 = attention_layer(self.conv3_1, self.conv3_3, self.layer2, bpool=True, reuse=reuse, name='od_att3')

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        conv = tf.nn.conv2d(self.pool3, self.kernel41, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases41)
        out = opt.batch_norm(out, name='vgg_bn41', training=trainable)
        self.conv4_1 = tf.nn.relu(out)

        # conv4_2
        conv = tf.nn.conv2d(self.conv4_1, self.kernel42, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases42)
        out = opt.batch_norm(out, name='vgg_bn42', training=trainable)
        self.conv4_2 = tf.nn.relu(out)

        # conv4_3
        conv = tf.nn.conv2d(self.conv4_2, self.kernel43, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases43)
        out = opt.batch_norm(out, name='vgg_bn43', training=trainable)
        self.conv4_3 = tf.nn.relu(out)

        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        self.layer4 = attention_layer(self.conv4_1, self.conv4_3, self.layer3, bpool=True, reuse=reuse, name='od_att4')

        # conv5_1
        conv = tf.nn.conv2d(self.pool4, self.kernel51, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases51)
        out = opt.batch_norm(out, name='vgg_bn51', training=trainable)
        self.conv5_1 = tf.nn.relu(out)

        # conv5_2
        conv = tf.nn.conv2d(self.conv5_1, self.kernel52, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases52)
        out = opt.batch_norm(out, name='vgg_bn52', training=trainable)
        self.conv5_2 = tf.nn.relu(out)

        # conv5_3
        conv = tf.nn.conv2d(self.conv5_2, self.kernel53, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.biases53)
        out = opt.batch_norm(out, name='vgg_bn53', training=trainable)
        self.conv5_3 = tf.nn.relu(out)

        self.layer5 = attention_layer(self.conv5_1, self.conv5_3, self.layer4, bpool=True, reuse=reuse, name='od_att5')

def get_dtype_np():
    return np.float32

def get_dtype_tf():
    return tf.float32

class squeezenet:
    def __init__(self, data_path):
        self.load_net(data_path)
        self.trainable = None

    def load_net(self, data_path):
        if not os.path.isfile(data_path):
            parser.error("Network %s does not exist. (Did you forget to download it?)" % data_path)

        weights_raw = scipy.io.loadmat(data_path)

        # Converting to needed type
        #conv_time = time.time()
        self.weights = {}
        for name in weights_raw:
            self.weights[name] = []
            # skipping '__version__', '__header__', '__globals__'
            if name[0:2] != '__':
                kernels, bias = weights_raw[name][0]
                self.weights[name].append(kernels.astype(get_dtype_np()))
                self.weights[name].append(bias.astype(get_dtype_np()))
        #print("Converted network data(%s): %fs" % (get_dtype_np(), time.time() - conv_time))

        #self.mean_pixel = np.array([104.006, 116.669, 122.679], dtype=get_dtype_np())

    def get_weights_biases(self, layer_name):
        weights, biases = self.weights[layer_name]
        biases = biases.reshape(-1)
        return (weights, biases)

    def _conv_layer(self, name, input, weights, bias, padding='SAME', stride=(1, 1)):
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, stride[0], stride[1], 1),
                            padding=padding)
        x = tf.nn.bias_add(conv, bias)
        return x

    def _act_layer(self, name, input):
        x = tf.nn.relu(input)
        return x

    def _pool_layer(self, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
        if pooling == 'avg':
            x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                               padding=padding)
        else:
            x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                               padding=padding)
        return x

    def fire_cluster(self, x, cluster_name):
        # central - squeeze
        layer_name = cluster_name + '/squeeze1x1'
        weights, biases = self.get_weights_biases(layer_name)
        x = self._conv_layer(layer_name + '_conv', x, weights, biases, padding='SAME')
        x = opt.batch_norm(x, name=cluster_name+'_bck_bn1', training=self.trainable)
        x = self._act_layer(layer_name + '_actv', x)

        # left - expand 1x1
        layer_name = cluster_name + '/expand1x1'
        weights, biases = self.get_weights_biases(layer_name)
        x_l = self._conv_layer(layer_name + '_conv', x, weights, biases, padding='SAME')
        x_l = opt.batch_norm(x_l, name=cluster_name+'_bck_bn2', training=self.trainable)
        x_l = self._act_layer(layer_name + '_actv', x_l)

        # right - expand 3x3
        layer_name = cluster_name + '/expand3x3'
        weights, biases = self.get_weights_biases(layer_name)
        x_r = self._conv_layer(layer_name + '_conv', x, weights, biases, padding='SAME')
        x_r = opt.batch_norm(x_r, name=cluster_name+'_bck_bnr', training=self.trainable)
        x_r = self._act_layer(layer_name + '_actv', x_r)

        # concatenate expand 1x1 (left) and expand 3x3 (right)
        x = tf.concat([x_l, x_r], 3)

        return x

    def forward(self, imgs, trainable):
        pooling = 'max'
        self.trainable = trainable

        mean = tf.constant([104.006, 116.669, 122.679], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        x = imgs - mean

        layer_name = 'conv1'
        weights, biases = self.get_weights_biases(layer_name)
        x = self._conv_layer(layer_name + '_conv', x, weights, biases, padding='SAME', stride=(2, 2))
        x = opt.batch_norm(x, name=layer_name+'_bck_bn', training=self.trainable)
        x = self._act_layer(layer_name + '_actv', x)
        x = self._pool_layer('pool1_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='SAME')

        self.fire2 = self.fire_cluster(x, cluster_name='fire2')
        self.fire3 = self.fire_cluster(self.fire2, cluster_name='fire3')
        x = self._pool_layer('pool3_pool', self.fire3, pooling, size=(3, 3), stride=(2, 2), padding='SAME')

        self.od_layer1 = attention_layer(self.fire2, self.fire3, None, bpool=False, reuse=False, name='od_att1')
        self.md_layer1 = attention_layer(self.fire2, self.fire3, None, bpool=False, reuse=False, name='md_att1')

        self.fire4 = self.fire_cluster(x, cluster_name='fire4')
        self.fire5 = self.fire_cluster(self.fire4, cluster_name='fire5')
        x = self._pool_layer('pool5_pool', self.fire5, pooling, size=(3, 3), stride=(2, 2), padding='SAME')

        self.od_layer2 = attention_layer(self.fire4, self.fire5, self.od_layer1, bpool=True, reuse=False, name='od_att2')
        self.md_layer2 = attention_layer(self.fire4, self.fire5, self.md_layer1, bpool=True, reuse=False, name='md_att2')

        self.fire6 = self.fire_cluster(x, cluster_name='fire6')
        self.fire7 = self.fire_cluster(self.fire6, cluster_name='fire7')
        self.fire8 = self.fire_cluster(self.fire7, cluster_name='fire8')
        self.fire9 = self.fire_cluster(self.fire8, cluster_name='fire9')

        self.od_layer3 = attention_layer(self.fire6, self.fire7, self.od_layer2, bpool=True, reuse=False, name='od_att3')
        self.od_layer4 = attention_layer(self.fire8, self.fire9, self.od_layer3, bpool=False, reuse=False, name='od_att4')

        self.md_layer3 = attention_layer(self.fire6, self.fire7, self.md_layer2, bpool=True, reuse=False, name='md_att3')
        self.md_layer4 = attention_layer(self.fire8, self.fire9, self.md_layer3, bpool=False, reuse=False, name='md_att4')

def attention_layer(layer1, layer2, player, bpool, reuse=False, name='attention'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        sz = layer1.get_shape()
        if player != None:
            if bpool == True:
                player = opt.maxpool2d(player)
            layer1 = tf.concat([layer1, player], axis=3)

        x = tf.concat([layer1, tf.reduce_mean(layer1, [3], keep_dims=True)], axis=3)
        x = opt.conv2d(x, 32, kernel=1, stride=1, name='conv1')
        x = opt.batch_norm(x, name='bn1', training=False)
        x = opt.prelu(x, name='pr1')

        x = opt.conv2d(x, sz[3], kernel=1, stride=1, name='conv2')
        x = tf.nn.sigmoid(x, name='pr2')

        x = layer2 * x

        x = opt.conv2d(x, sz[3], kernel=1, stride=1, name='conv3')
        x = opt.batch_norm(x, name='bn3', training=False)
        x = opt.prelu(x, name='pr3')

        return x

def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data




