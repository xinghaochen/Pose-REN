import cv2
import numpy as np
import os

import caffe
from caffe.proto import caffe_pb2
import util

class ModelPoseREN(object):
    def __init__(self, dataset, center_loader=None,
            param=None, use_gpu=False):
        self._dataset = dataset
        self._center_loader = center_loader
        init_proto_name, init_model_name = util.get_model(dataset, 'baseline')
        proto_name, model_name = util.get_model(dataset, 'pose_ren')

        self._fx, self._fy, self._ux, self._uy = util.get_param(dataset) if param is None else param
        self._net = caffe.Net(proto_name, caffe.TEST, weights=model_name)
        self._net_init = caffe.Net(init_proto_name, caffe.TEST, weights=init_model_name)

        self._input_size = self._net.blobs['data'].shape[-1]
        self._cube_size = 150
        if use_gpu:
            caffe.set_mode_gpu()
            caffe.set_device(0)
        else:
            caffe.set_mode_cpu()


    def reset_model(self, dataset, test_id = 0):
        self._dataset = dataset
        if dataset == 'msra':
            init_proto_name, init_model_name = util.get_model(dataset, 'baseline', test_id)
            proto_name, model_name = util.get_model(dataset, 'pose_ren', test_id)
        else:
            init_proto_name, init_model_name = util.get_model(dataset, 'baseline')
            proto_name, model_name = util.get_model(dataset, 'pose_ren')

        print init_proto_name, init_model_name
        print proto_name, model_name
        self._net = caffe.Net(proto_name, caffe.TEST, weights=model_name)
        self._net_init = caffe.Net(init_proto_name, caffe.TEST, weights=init_model_name)



    def detect_images(self, imgs, centers=None):
        assert centers is not None or self._center_loader is not None
        batch_size = len(imgs)
        if centers is None:
            centers = np.zeros([batch_size, 3], dtype=np.float32)
            for idx, img in enumerate(imgs):
                centers[idx, :] = self._center_loader(img)
        _, channels, height, width = self._net.blobs['data'].shape
        # run Init-CNN
        self._net_init.blobs['data'].reshape(batch_size, channels, height, width)
        cropped_images = []
        for idx in range(batch_size):
            cropped_image = self._crop_image(imgs[idx], centers[idx])
            cropped_images.append(cropped_image)
            self._net_init.blobs['data'].data[idx, ...] = cropped_image
        if self._dataset == 'hands17':
            init_poses = self._net_init.forward()['predict']
        else:
            init_poses = self._net_init.forward()['fc3']
        # run Pose-REN
        prev_pose = init_poses

        self._net.blobs['data'].reshape(batch_size, channels, height, width)
        _, channels = self._net.blobs['prev_pose'].shape
        self._net.blobs['prev_pose'].reshape(batch_size, channels)
        for idx in range(batch_size):
            self._net.blobs['data'].data[idx, ...] = cropped_images[idx]
        for it in xrange(3):
            self._net.blobs['prev_pose'].data[...] = prev_pose
            if self._dataset == 'hands17':
                poses = self._net.forward()['predict']
            else:
                poses = self._net.forward()['fc3_0']
            prev_pose = poses
        return self._transform_pose(poses, centers), cropped_images
    
    def detect_image(self, img, center=None):
        if center is None:
            res, cropped_image = self.detect_images([img])
        else:
            res, cropped_image = self.detect_images([img], [center])
        return res[0, ...], cropped_image[0]

    def detect_files(self, base_dir, names, centers=None, dataset=None, max_batch=64, is_flip=False):
        assert max_batch > 0
        if dataset is None:
            dataset = self._dataset

        batch_imgs = []
        batch_centers = []
        results = []
        for idx, name in enumerate(names):
            img = util.load_image(dataset, os.path.join(base_dir, name),
                    is_flip=is_flip)
            batch_imgs.append(img)
            if centers is None:
                batch_centers.append(self._center_loader(img))
            else:
                batch_centers.append(centers[idx, :])

            if len(batch_imgs) == max_batch:
                res, _= self.detect_images(batch_imgs, batch_centers)
                for line in res:
                    results.append(line)
                del batch_imgs[:]
                del batch_centers[:]
                print('{}/{}'.format(idx + 1, len(names)))
        if batch_imgs:
            res, _ = self.detect_images(batch_imgs, batch_centers)
            for line in res:
                results.append(line)
        print('done!')
        return np.array(results)
    
    def _crop_image(self, img, center, is_debug=False):
        xstart = center[0] - self._cube_size / center[2] * self._fx
        xend = center[0] + self._cube_size / center[2] * self._fx
        ystart = center[1] - self._cube_size / center[2] * self._fy
        yend = center[1] + self._cube_size / center[2] * self._fy

        src = [(xstart, ystart), (xstart, yend), (xend, ystart)]    
        dst = [(0, 0), (0, self._input_size - 1), (self._input_size - 1, 0)]
        trans = cv2.getAffineTransform(np.array(src, dtype=np.float32),
                np.array(dst, dtype=np.float32))
        res_img = cv2.warpAffine(img, trans, (self._input_size, self._input_size), None,
                cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, center[2] + self._cube_size)
        res_img -= center[2]
        res_img = np.maximum(res_img, -self._cube_size)
        res_img = np.minimum(res_img, self._cube_size)
        res_img /= self._cube_size

        if is_debug:
            img_show = (res_img + 1) / 2;
            hehe = cv2.resize(img_show, (512, 512))
            cv2.imshow('debug', img_show)
            ch = cv2.waitKey(0)
            if ch == ord('q'):
                exit(0)

        return res_img

    def _transform_pose(self, poses, centers):
        res_poses = np.array(poses) * self._cube_size
        num_joint = poses.shape[1] / 3
        centers_tile = np.tile(centers, (num_joint, 1, 1)).transpose([1, 0, 2])
        res_poses[:, 0::3] = res_poses[:, 0::3] * self._fx / centers_tile[:, :, 2] + centers_tile[:, :, 0]
        res_poses[:, 1::3] = res_poses[:, 1::3] * self._fy / centers_tile[:, :, 2] + centers_tile[:, :, 1]
        res_poses[:, 2::3] += centers_tile[:, :, 2]
        res_poses = np.reshape(res_poses, [poses.shape[0], -1, 3])
        if self._dataset == 'nyu':
            res_poses = res_poses[:, [6, 7, 8, 9, 10, 11, 12, 13, 3, 4, 5, 1, 2, 0], :]
        return res_poses
