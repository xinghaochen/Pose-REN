'''
Hengkai Guo: https://github.com/guohengkai/region-ensemble-network/blob/master/evaluation/util.py
Modified by Xinghao Chen
Apr. 2018
'''

import cv2
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) # src
sys.path.append(ROOT_DIR) # config
# sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
from enum import Enum


def get_positions(in_file):
    with open(in_file) as f:
        positions = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(positions), (-1, len(positions[0]) / 3, 3))


def check_dataset(dataset):
    return dataset in set(['icvl', 'nyu', 'msra', 'hands17'])


def get_dataset_file(dataset):
    return 'labels/{}_test_label.txt'.format(dataset)


def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 588.03, 587.07, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120
    elif dataset == 'hands17':
        return 475.065948, 475.065857, 315.944855, 245.287079

def get_joint_num(dataset):
    joint_num_dict = {'nyu': 14, 'icvl': 16, 'msra': 21, 'hands17': 21}
    return joint_num_dict[dataset]

def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def get_errors(dataset, in_file):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    labels = get_positions(get_dataset_file(dataset))
    outputs = get_positions(in_file)
    params = get_param(dataset)
    labels = pixel2world(labels, *params)
    outputs = pixel2world(outputs, *params)
    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))
    return errors


def get_model(dataset, name='baseline', test_id = 0):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    if dataset == 'hands17':
        if name == 'baseline':
            return (os.path.join(ROOT_DIR, '../models/deploy_{}_{}.prototxt'.format(dataset, name)),
                    os.path.join(ROOT_DIR, '../models/model_{}_{}_full_large_aug.caffemodel'.format(name, dataset, test_id)))
        elif name == 'pose_ren':
            return (os.path.join(ROOT_DIR, '../models/deploy_{}_{}_py.prototxt'.format(dataset, name)),
                    os.path.join(ROOT_DIR, '../models/model_{}_{}_full_large_aug.caffemodel'.format(name, dataset, test_id)))
    elif dataset == 'msra':
        return (os.path.join(ROOT_DIR, '../models/deploy_{}_{}.prototxt'.format(dataset, name)),
                os.path.join(ROOT_DIR, '../models/{}_{}_{}.caffemodel'.format(dataset, name, test_id)))
    else:
        return (os.path.join(ROOT_DIR, '../models/deploy_{}_{}.prototxt'.format(dataset, name)),
                os.path.join(ROOT_DIR, '../models/{}_{}.caffemodel'.format(dataset, name)))

def read_depth_from_bin(image_name):
    f = open(image_name, 'rb')
    data = np.fromfile(f, dtype=np.uint32)
    width, height, left, top, right , bottom = data[:6]
    depth = np.zeros((height, width), dtype=np.float32)
    f.seek(4*6)
    data = np.fromfile(f, dtype=np.float32)
    depth[top:bottom, left:right] = np.reshape(data, (bottom-top, right-left))
    return depth


def load_image(dataset, name, input_size=None, is_flip=False):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    if dataset == 'icvl':
        img = cv2.imread(name, 2)  # depth image
        img[img == 0] = img.max()  # invalid pixel
        img = img.astype(float)
    elif dataset == 'nyu':
        img = cv2.imread(name)
        g = np.asarray(img[:, :, 1], np.int32)
        b = np.asarray(img[:, :, 0], np.int32)
        dpt = np.bitwise_or(np.left_shift(g, 8), b)
        img = np.asarray(dpt, np.float32)
    elif dataset == 'msra':
        img = read_depth_from_bin(name)
        img[img == 0] = 10000

    if input_size is not None:
        img = cv2.resize(img, (input_size, input_size))
    if is_flip:
        img[:, ::-1] = img
    return img


def load_names(dataset):
    with open('{}/results/{}_test_list.txt'.format(os.path.join(ROOT_DIR, '..'), dataset)) as f:
        return [line.strip() for line in f]


def load_centers(dataset):
    with open('{}/results/{}_center.txt'.format(os.path.join(ROOT_DIR, '..'), dataset)) as f:
        return np.array([map(float,
            line.strip().split()) for line in f])


def get_sketch_setting(dataset):
    if dataset == 'icvl':
        return [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15)]
    elif dataset == 'nyu':
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10), (1, 13),
                (3, 13), (5, 13), (7, 13), (10, 13), (11, 13), (12, 13)]
    elif dataset == 'msra':
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
    elif dataset == 'hands17':
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11), (3, 12), (12, 13), (13, 14), (4, 15), (15, 16),
                (16, 17), (5, 18), (18, 19), (19, 20)]

class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (17, 240, 244)
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)


def get_sketch_color(dataset):
    if dataset == 'icvl':
        return [Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.BLUE, Color.GREEN,
                Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED,
              Color.GREEN, Color.GREEN, Color.GREEN,
              Color.BLUE, Color.BLUE, Color.BLUE,
              Color.YELLOW, Color.YELLOW, Color.YELLOW,
              Color.PURPLE, Color.PURPLE, Color.PURPLE,
              Color.RED, Color.RED, Color.RED]

def get_joint_color(dataset):
    if dataset == 'icvl':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.GREEN, Color.RED, Color.RED, Color.PURPLE, Color.PURPLE, Color.YELLOW, Color.YELLOW,
                Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN, Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.CYAN, Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.PURPLE, Color.PURPLE, Color.PURPLE,
                Color.RED, Color.RED, Color.RED]

def draw_pose_old(dataset, img, pose):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), (0, 0, 255), 1)
    return img


def draw_pose(dataset, img, pose):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    colors = get_sketch_color(dataset)
    colors_joint = get_joint_color(dataset)
    idx = 0
    #plt.figure()
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, colors_joint[idx].value, -1)
        #plt.scatter(pt[0], pt[1], pt[2])
        idx = idx + 1
    idx = 0
    for x, y in get_sketch_setting(dataset):
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 2)
        idx = idx + 1
    #plt.show()
    return img


def get_center(img, upper=650, lower=1):
    centers = np.array([0.0, 0.0, 300.0])
    count = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] <= upper and img[y, x] >= lower:
                centers[0] += x
                centers[1] += y
                centers[2] += img[y, x]
                count += 1
    if count:
        centers /= count
    return centers


def get_center_fast(img, upper=650, lower=1):
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    #print centers
    return centers


def save_results(results, out_file):
    with open(out_file, 'w') as f:
        for result in results:
            for j in range(result.shape[0]):
                for k in range(result.shape[1]):
                    f.write('{:.3f} '.format(result[j, k]))
            f.write('\n')
