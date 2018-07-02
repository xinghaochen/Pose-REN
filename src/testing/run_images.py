import argparse

import util

from src.utils.model_pose_ren import HandModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_model', help='the dataset type for model')
    parser.add_argument('model_prefix', help='the model prefix')
    parser.add_argument('base_dir', help='the base directory for image')
    parser.add_argument('in_file', default=None, help='input image list')
    parser.add_argument('out_file', default=None, help='output file for pose')
    parser.add_argument('--dataset_image', default=None,
            help='the dataset type for loading images, use the same as dataset_model when empty')
    parser.add_argument('--is_flip', action='store_true', help='flip the input')
    parser.add_argument('--upper', type=int, default=700, help='upper value for segmentation')
    parser.add_argument('--lower', type=int, default=1, help='lower value for segmentation')
    parser.add_argument('--fx', type=float, default=371.62, help='fx')
    parser.add_argument('--fy', type=float, default=370.19, help='fy')
    parser.add_argument('--ux', type=float, default=256, help='ux')
    parser.add_argument('--uy', type=float, default=212, help='uy')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_model = args.dataset_model
    dataset_image = args.dataset_image
    if dataset_image is None:
        dataset_image = dataset_model

    hand_model = HandModel(dataset_model, args.model_prefix,
            lambda img: util.get_center(img, lower=args.lower, upper=args.upper),
            param=(args.fx, args.fy, args.ux, args.uy))
    with open(args.in_file) as f:
        names = [line.strip() for line in f]
    results = hand_model.detect_files(args.base_dir, names, dataset=dataset_image,
            is_flip=args.is_flip)
    util.save_results(results, args.out_file)


if __name__ == '__main__':
    main()
