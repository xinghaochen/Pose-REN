'''
Hengkai Guo: https://github.com/guohengkai/region-ensemble-network
Modified by Xinghao Chen
Apr. 2018
'''

import argparse
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)  # config
sys.path.append(os.path.join(ROOT_DIR, 'utils'))  # utils

import cv2
import numpy as np

from utils import util


def show_pose(dataset_model, dataset_image, base_dir, outputs, list_file, save_dir,
        is_flip, gif):
    if list_file is None:
        names = util.load_names(dataset_image)
    else:
        with open(list_file) as f:
            names = [line.strip() for line in f]
    assert len(names) == outputs.shape[0]

    for idx, (name, pose) in enumerate(zip(names, outputs)):
        img = util.load_image(dataset_image, os.path.join(base_dir, name),
                              is_flip=is_flip)
        img = img.astype(np.float32)
        # img[img >= 1000] = 1000
        img = (img - img.min())*255 / (img.max() - img.min())
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = util.draw_pose(dataset_model, img, pose)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '{}/{}'.format(idx, len(names)), (20, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('result', img)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:>06d}.png'.format(idx)), img)
        ch = cv2.waitKey(25)
        if ch == ord('q'):
            break

    if gif and save_dir is not None:
        os.system('convert -loop 0 -page +0+0 -delay 25 {0}/*.png {0}/output.gif'.format(save_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_model', help='the dataset type for model')
    parser.add_argument('base_dir', help='the base directory for image')
    parser.add_argument('--in_file', default=None, help='input file for pose, empty for using labels')
    parser.add_argument('--save_dir', default=None, help='save directory for image')
    parser.add_argument('--dataset_image', default=None,
            help='the dataset type for loading images, use the same as dataset_model when empty')
    parser.add_argument('--gif', action='store_true', help='save gif')
    parser.add_argument('--list_file', default=None, help='input image list')
    parser.add_argument('--is_flip', action='store_true', help='flip the input')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_model = args.dataset_model
    dataset_image = args.dataset_image
    if dataset_image is None:
        dataset_image = dataset_model
    if args.in_file is None:
        print('no input file, using ground truth')
        in_file = util.get_dataset_file(dataset_image)
    else:
        in_file = args.in_file

    save_dir = args.save_dir
    if save_dir is not None and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    outputs = util.get_positions(in_file)
    show_pose(dataset_model, dataset_image, args.base_dir,
            outputs, args.list_file, save_dir, args.is_flip, args.gif)


if __name__ == '__main__':
    main()
