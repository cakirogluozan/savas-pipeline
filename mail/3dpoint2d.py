
import numpy as np
import cv2

def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def compute_box_3d(ry, l1, w1, h1, t0, t1, t2):

    R = roty(ry)

    l = w1
    w = l1
    h = h1

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + t0
    corners_3d[1, :] = corners_3d[1, :] + t1
    corners_3d[2, :] = corners_3d[2, :] + t2

    return corners_3d


def project_to_image(ppoints, v2irot, v2itr, camera_matrix, dist_coeffs):
    points, jac = cv2.projectPoints(ppoints, v2irot, v2itr, camera_matrix, dist_coeffs)
    points = np.transpose(np.reshape(points, (points.shape[0], 2)))

    return points

def project2d_boundary(tvl):
    xmin = tvl[0, 0]
    xmax = tvl[0, 0]
    ymin = tvl[1, 0]
    ymax = tvl[1, 0]

    for i in range(1, tvl.shape[1]):
        #print(tvl[0,i], tvl[1,i], xmin, ymin, xmax, ymax)
        if xmin > tvl[0,i]:
            xmin = tvl[0,i]

        if ymin > tvl[1,i]:
            ymin = tvl[1,i]

        if xmax < tvl[0,i]:
            xmax = tvl[0,i]

        if ymax < tvl[1,i]:
            ymax = tvl[1,i]

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)

    xmax = min(xmax, 1920-1)
    ymax = min(ymax, 1208-1)

    return int(xmin), int(ymin), int(xmax), int(ymax)

l2crot = np.load('l2crot.npy')
l2ctr = np.load('l2ctr.npy')
cam_matrix = np.load('cam_matrix.npy')
dist_coeff = np.load('dist_coeff.npy')

### Read head, r1, r2, r3, m1, m2, m3 parameters from JSON file from 3d labels

tvl = compute_box_3d(head, r1, r2, r3, m1, m2, m3)  # [8,3]
tvl2d = project_to_image(tvl, l2crot, l2ctr, cam_matrix, dist_coeff)  # [8,2]
box2d = project2d_boundary(tvl2d)