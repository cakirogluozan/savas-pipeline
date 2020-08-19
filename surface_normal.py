import os

import matplotlib.pyplot as plt
import numpy as np
from utils import *

import open3d as o3d
import tensorflow as tf

import pickle

def load_image(im):
    im = cv2.resize(im, (512, 256))
    im_str = cv2.imencode('.jpg', im)[1].tostring()
    return im_str

def load_binary(im):
    im = cv2.resize(im, (512, 256))
    im_str = np.reshape(im, (512*256*3))
    return im_str

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def make_tfrecord(img2, img3, surface, mask, path):

    feature = {
        'left': _bytes_feature([load_image(img2)]),
        'right': _bytes_feature([load_image(img3)]),
        'surface': _float_feature(load_binary(surface)),
        'mask': _bytes_feature([load_image(mask)])
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=feature))

    with tf.python_io.TFRecordWriter(path) as writer:
        writer.write(example.SerializeToString())

def make_picklerecord(pthfile, surface, mask):
    data = [surface, mask]
    with open(pthfile, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

def render_lidar_on_image(pts_velo, calib, img_width, img_height, pthimage2, pthimage3, pthrecord):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

    lidar = o3d.geometry.PointCloud()
    lidar.points = o3d.utility.Vector3dVector( pts_velo[inds, :])
    lidar.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=1, max_nn=10))
    normal = np.asarray(lidar.normals)

    surface = np.zeros((img_height, img_width, 3))
    mask = np.zeros((img_height, img_width, 1))
    for i in range(imgfov_pc_pixel.shape[1]):
        xid = int(np.round(imgfov_pc_pixel[0, i]))
        yid = int(np.round(imgfov_pc_pixel[1, i]))

        if xid == img_width or yid == img_height:
            continue

        surface[yid, xid, :] = normal[i, :]
        mask[yid, xid, 0] = 255.

    #im2 = cv2.imread(pthimage2)
    #im3 = cv2.imread(pthimage3)

    #print(np.max(surface), np.min(surface))
    #make_record(im2, im3, surface, mask, pthrecord)
    make_picklerecord(pthrecord, surface, mask)

import glob

if __name__ == '__main__':

    image2 = sorted(glob.glob('/data1/savas/dataset/kitti_object/training/image_2/*.png'))
    image3 = sorted(glob.glob('/data1/savas/dataset/kitti_object/training/image_3/*.png'))
    calib  = sorted(glob.glob('/data1/savas/dataset/kitti_object/training/calib/*.txt'))
    lidar  = sorted(glob.glob('/data1/savas/dataset/kitti_object/training/velodyne/*.bin'))

    for idx, pthcalib in enumerate(calib):
        print(os.path.basename(image2[idx]))
        pthrecord = '/data1/savas/dataset/kitti_object/training/surface/' + os.path.basename(image2[idx].split('.')[0] + '.pickle')
        if os.path.exists(pthrecord):
            continue

        rgb = cv2.cvtColor(cv2.imread(image2[idx]), cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = rgb.shape

        calib = read_calib_file(pthcalib)
        pc_velo = load_velo_scan(lidar[idx])[:, :3]

        render_lidar_on_image(pc_velo, calib, img_width, img_height, image2[idx], image3[idx], pthrecord)
