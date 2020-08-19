
import numpy as np
import os, glob

basedir = '/home/sozkan12/datasets/kitti_object/training/'
imlist = sorted(glob.glob(basedir + 'image_2/*.png'))
rightlist = sorted(glob.glob(basedir + 'image_3/*.png'))
lblist = sorted(glob.glob(basedir + 'label_2/*.txt'))

assert(len(imlist) == len(lblist))

fileout = open('data/dataset/kitti_train_stereo.txt', 'w')
classids = {'Pedestrian': 0, 'Car': 1, 'Tram': 2, 'Van': 3, 'Truck': 4, 'Cyclist': 5}

gtlines = []
for idx in range(len(imlist)):
    filein = open(lblist[idx], 'r')
    lines = filein.readlines()

    gtlabel = imlist[idx]
    gtlabel += ' ' + rightlist[idx]
    for ln in lines:
        gtlabel += ' '
        ln = ln.split(' ')
        type = ln[0]
        xmin = int(float(ln[4]))
        ymin = int(float(ln[5]))
        xmax = int(float(ln[6]))
        ymax = int(float(ln[7]))

        typeidx = None
        try:
            typeidx = classids[type]
        except:
            continue

        gtlabel += str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) +  ',' + str(typeidx)

    gtlabel += '\n'
    gtlines.append(gtlabel)

fileout.writelines(gtlines)
fileout.close()
filein.close()
