#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg
import core.backbone as backbone

input_data = tf.placeholder(dtype=tf.float32, name='input_data')
trainable    = tf.placeholder(dtype=tf.bool, name='training')

backbone = backbone.squeezenet('sqz_full.mat')
backbone.forward(input_data, trainable)

def load_weights(parameters, weight_file, sess):
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    for i, k in enumerate(keys):
        if i > 25:
            continue

        print(i, k, np.shape(weights[k]))
        sess.run(parameters[i].assign(weights[k]))

def train(yolomodel, monomodel, trainset, testset):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    yolomodel.first_stage_epochs = 0

    with tf.name_scope('loader_and_saver'):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    #load_weights(backbone.parameters, '/home/sozkan12/PycharmProjects/lanedetection/vgg16_weights.npz', sess)

    for epoch in range(1, 1 + yolomodel.first_stage_epochs + yolomodel.second_stage_epochs):
        if epoch <= yolomodel.first_stage_epochs:
            train_op_yolo = yolomodel.train_op_with_frozen_variables
            train_op_mono = monomodel.train_op_with_all_variables
        else:
            train_op_yolo = yolomodel.train_op_with_all_variables
            train_op_mono = monomodel.train_op_with_all_variables

        pbar = tqdm(trainset)
        vtrain_yolo_loss, vtest_yolo_loss, vtrain_mono_loss, vtest_mono_loss = [], [], [], []

        for train_data in pbar:
            _, _, summary, train_yolo_loss, train_mono_loss, global_step_val = sess.run(
                [train_op_yolo, train_op_mono, yolomodel.write_op, yolomodel.loss, monomodel.total_loss, yolomodel.global_step], feed_dict={
                    input_data: train_data[0],
                    monomodel.image_left: train_data[0],
                    monomodel.image_right: train_data[1],
                    yolomodel.label_sbbox: train_data[2],
                    yolomodel.label_mbbox: train_data[3],
                    yolomodel.label_lbbox: train_data[4],
                    yolomodel.true_sbboxes: train_data[5],
                    yolomodel.true_mbboxes: train_data[6],
                    yolomodel.true_lbboxes: train_data[7],
                    trainable: True,
                })

            vtrain_yolo_loss.append(train_yolo_loss)
            vtrain_mono_loss.append(train_mono_loss)
            #yoloclass.summary_writer.add_summary(summary, global_step_val)
            pbar.set_description("train loss: %.2f, %.2f" % (train_yolo_loss, train_mono_loss))

        for test_data in testset:
            test_yolo_loss, test_mono_loss = sess.run([yolomodel.loss, monomodel.total_loss], feed_dict={
                    input_data: test_data[0],
                    monomodel.image_left: test_data[0],
                    monomodel.image_right: test_data[1],
                    yolomodel.label_sbbox: test_data[2],
                    yolomodel.label_mbbox: test_data[3],
                    yolomodel.label_lbbox: test_data[4],
                    yolomodel.true_sbboxes: test_data[5],
                    yolomodel.true_mbboxes: test_data[6],
                    yolomodel.true_lbboxes: test_data[7],
                    trainable: False,
                })

            vtest_yolo_loss.append(test_yolo_loss)
            vtest_mono_loss.append(test_mono_loss)

        train_yolo_loss, test_yolo_loss, train_mono_loss, test_mono_loss = np.mean(vtrain_yolo_loss), np.mean(vtest_yolo_loss), np.mean(vtrain_mono_loss), np.mean(vtest_mono_loss)
        ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_yolo_loss
        log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("YOLO=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
              % (epoch, log_time, train_yolo_loss, test_yolo_loss, ckpt_file))
        print("MONO=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
              % (epoch, log_time, train_mono_loss, test_mono_loss, ckpt_file))
        saver.save(sess, ckpt_file, global_step=epoch)


class YoloTrain(object):
    def __init__(self, steps_per_period):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)

        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.steps_per_period    = steps_per_period

        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS

        self.max_bbox_per_scale  = 150


        with tf.name_scope('define_input'):
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3() #(self.input_data, self.trainable)

            self.model.forward(backbone, trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []

            varlist = tf.trainable_variables()
            od_vars = [var for var in varlist if 'od_' in var.name]
            for var in  tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['od_conv_sbbox', 'od_conv_mbbox', 'od_conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            varlist = tf.trainable_variables()
            od_vars = [var for var in varlist if 'od_' in var.name]
            bck_vars = [var for var in varlist if 'bck_' in var.name]

            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=od_vars)

            second_stage_optimizerbc = tf.train.AdamOptimizer(self.learn_rate).minimize(0.1*self.loss,
                                                                                      var_list=bck_vars)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, second_stage_optimizerbc, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./data/log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            #self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)

from core.bilinear_sampler import *
import core.opt as opt
import tensorflow.contrib.slim as slim

class MonoDepthTrain():
    def __init__(self, steps_per_period):
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.steps_per_period    = steps_per_period

        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS

        with tf.name_scope('define_input'):
            self.image_left  = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 3], name='image_left')
            self.image_right = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 3], name='image_right')

            self.left_pyramid = self.scale_pyramid(self.image_left, 3)
            self.right_pyramid = self.scale_pyramid(self.image_right, 3)

        self.alpha_image_loss = 0.85
        self.disp_gradient_loss_weight = 0.1
        self.lr_loss_weight = 1.0

        with tf.name_scope("forward"):
            self.forward(backbone, trainable)

        with tf.name_scope("define_loss"):
            self.disparity_load()

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("train_stage"):
            varlist = tf.trainable_variables()
            od_vars = [var for var in varlist if 'md_' in var.name]
            bck_vars = [var for var in varlist if 'bck_' in var.name]

            train_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss,
                                                      var_list=od_vars)

            train_stage_optimizerbc = tf.train.AdamOptimizer(self.learn_rate).minimize(0.1*self.total_loss,
                                                                                      var_list=bck_vars)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([train_stage_optimizer, train_stage_optimizerbc, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(3)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(3)]
        return smoothness_x + smoothness_y

    def disparity_load(self):
        with tf.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3]
            self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]

        with tf.variable_scope('images'):
            self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(3)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(3)]

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(3)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(3)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness  = self.get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

        self.build_losses()

    def forward(self, backbone, trainable=trainable, name='md_disparity', reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            xin = backbone.md_layer4

            x = opt.conv2d(xin, 512, kernel=3, stride=1, name='conv4')
            x = opt.prelu(x, name='pr4')

            x = tf.image.resize_images(x, [64, 64])
            xin = backbone.md_layer2

            x = tf.concat([x, xin], axis=3)
            x = opt.conv2d(x, 512, kernel=3, stride=1, name='conv3')
            x = opt.prelu(x, name='pr3')

            self.disp3 = opt.conv2d(x, 2, kernel=3, stride=1, name='disp3')
            self.disp3 = 0.3 * tf.nn.sigmoid(self.disp3)

            x = tf.concat([x, self.disp3], axis=3)
            x = tf.image.resize_images(x, [128, 128])
            xin = backbone.md_layer1  # , name='pre3', training=btrain)

            x = tf.concat([x, xin], axis=3)
            x = opt.conv2d(x, 512, kernel=3, stride=1, name='conv2')
            x = opt.prelu(x, name='pr2')

            self.disp2 = opt.conv2d(x, 2, kernel=3, stride=1, name='disp2')
            self.disp2 = 0.3 * tf.nn.sigmoid(self.disp2)

            x = tf.concat([x, self.disp2], axis=3)
            x = tf.image.resize_images(x, [512, 512])
            x = opt.conv2d(x, 64, kernel=3, stride=1, name='conv1')
            x = opt.prelu(x, name='pr1')

            self.disp1 = opt.conv2d(x, 2, kernel=3, stride=1, name='disp1')
            self.disp1 = 0.3 * tf.nn.sigmoid(self.disp1)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=False):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(3)]
            self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(3)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(3)]
            self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(3)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [self.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(3)]
            self.image_loss_left  = [self.alpha_image_loss * self.ssim_loss_left[i]  + (1 - self.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(3)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss  = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  / 8 ** i for i in range(3)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 8 ** i for i in range(3)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i]))  for i in range(3)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(3)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # TOTAL LOSS
            self.total_loss = self.image_loss + self.disp_gradient_loss_weight * self.disp_gradient_loss + self.lr_loss_weight * self.lr_loss

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]

        def scale_func(ratio):
            nh = h // ratio
            nw = w // ratio
            im = tf.image.resize_area(img, [nh, nw])

            return im

        scaled_imgs.append(scale_func(4))
        scaled_imgs.append(scale_func(8))

        return scaled_imgs

if __name__ == '__main__':

    trainset = Dataset('train')
    testset = Dataset('test')

    yolomodel = YoloTrain(len(trainset))
    monomodel = MonoDepthTrain(len(trainset))
    train(yolomodel, monomodel, trainset, testset)




