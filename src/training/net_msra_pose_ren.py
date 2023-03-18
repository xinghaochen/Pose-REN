import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))  # config
import config
import caffe
from caffe import layers as L, params as P
import util
from caffe_util import *


def pose_ren_net(net_type, iter_idx, output_dir, test_id = 0):
    dataset = 'msra'
    n = caffe.NetSpec()
    fx_, fy_, ux_, uy_ = util.get_param(dataset)
    point_num_ = util.get_joint_num(dataset)
    root_folder_ = config.msra_data_dir
    # data layers
    if net_type == 'train':
        image_source_ = '{}/cache/train_image_{}_s{}.txt'.format(output_dir, test_id, iter_idx)
        label_source_ = '{}/cache/train_label_{}_s{}.txt'.format(output_dir, test_id, iter_idx)
        pose_data_param_train = dict(image_source=image_source_,
                                     label_source=label_source_,
                                     root_folder=root_folder_,
                                     batch_size=128, shuffle=True, new_height=96, new_width=96,
                                     point_num=point_num_ * 2, point_dim=3,
                                     cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.MSRA)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=0),
                                     transform_param=dict(is_trans=True, trans_dx=10, trans_dy=10,
                                                          is_rotate=True, rotate_deg=180, is_zoom=True, zoom_scale=0.1),
                                     pose_data_param=pose_data_param_train, ntop=2)
        n.pose, n.prev_pose = L.Slice(n.label, slice_param=dict(slice_dim=1, slice_point=point_num_ * 3),
                                      include=dict(phase=0), ntop=2)
        first_layer = str(n.to_proto())

        pose_data_param_test = dict(image_source='{}/cache/test_image_{}.txt'.format(output_dir, test_id),
                                    label_source='{}/cache/test_label_{}_s{}.txt'.format(output_dir, test_id, iter_idx),
                                    root_folder=root_folder_,
                                    batch_size=128, shuffle=False, new_height=96, new_width=96,
                                    point_num=point_num_ * 2, point_dim=3, output_center=True,
                                    cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.MSRA)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=1),
                                     transform_param=dict(is_trans=False, is_rotate=False, is_zoom=False),
                                     pose_data_param=pose_data_param_test, ntop=2)
        n.pose, n.prev_pose, n.center = L.Slice(n.label, slice_param=dict(slice_dim=1,
                                                                          slice_point=[point_num_ * 3, point_num_ * 6]),
                                                include=dict(phase=1), ntop=3)
    elif net_type == 'test-train':
        label_source_ = '{}/cache/train_label_{}_s{}_single.txt'.format(output_dir, test_id, iter_idx)
        pose_data_param_test = dict(image_source='{}/cache/train_image_{}.txt'.format(output_dir, test_id),
                                    label_source=label_source_,
                                    root_folder=root_folder_,
                                    batch_size=128, shuffle=False, new_height=96, new_width=96,
                                    point_num=point_num_ * 2, point_dim=3, output_center=True,
                                    cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.MSRA)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=1),
                                     transform_param=dict(is_trans=False, is_rotate=False, is_zoom=False),
                                     pose_data_param=pose_data_param_test, ntop=2)
        n.pose, n.prev_pose, n.center = L.Slice(n.label, slice_param=dict(slice_dim=1,
                                                                          slice_point=[point_num_ * 3, point_num_ * 6]),
                                                include=dict(phase=1), ntop=3)
    elif net_type == 'test-test':
        label_source_ = '{}/cache/test_label_{}_s{}.txt'.format(output_dir, test_id, iter_idx)
        pose_data_param_test = dict(image_source='{}/cache/test_image_{}.txt'.format(output_dir, test_id),
                                    label_source=label_source_,
                                    root_folder=root_folder_,
                                    batch_size=128, shuffle=False, new_height=96, new_width=96,
                                    point_num=point_num_ * 2, point_dim=3, output_center=True,
                                    cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.MSRA)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=1),
                                     transform_param=dict(is_trans=False, is_rotate=False, is_zoom=False),
                                     pose_data_param=pose_data_param_test, ntop=2)
        n.pose, n.prev_pose, n.center = L.Slice(n.label, slice_param=dict(slice_dim=1,
                                                                          slice_point=[point_num_ * 3, point_num_ * 6]),
                                                include=dict(phase=1), ntop=3)

    # the base net
    n.conv0, n.relu0 = conv_relu(n.data, 16)
    n.conv1 = conv(n.relu0, 16)
    n.pool1 = max_pool(n.conv1)
    n.relu1 = L.ReLU(n.pool1, in_place=True)

    n.conv2_0, n.relu2_0 = conv_relu(n.pool1, 32, ks=1, pad=0)
    n.conv2, n.relu2 = conv_relu(n.relu2_0, 32)
    n.conv3 = conv(n.relu2, 32)
    n.res1 = L.Eltwise(n.conv2_0, n.conv3)
    n.pool2 = max_pool(n.res1)
    n.relu3 = L.ReLU(n.pool2, in_place=True)

    n.conv3_0, n.relu3_0 = conv_relu(n.relu3, 64, ks=1, pad=0)
    n.conv4, n.relu4 = conv_relu(n.relu3_0, 64)
    n.conv5 = conv(n.relu4, 64)
    n.res2 = L.Eltwise(n.conv3_0, n.conv5)
    n.pool3 = max_pool(n.res2)
    n.relu5 = L.ReLU(n.pool3, in_place=True)

    # pose guided region ensemble
    for idx in xrange(point_num_):
        if (idx-1) % 4 == 0 or (idx-3) % 4 == 0:
            continue
        rois = 'rois_{}'.format(idx)
        n[rois] = L.Python(n.prev_pose, module='python_layers.py_generate_roi_layer',
                           layer='PyGenerateROILayer', ntop=1,
                           param_str=str(dict(joint_idx=idx, roi_h=6, roi_w=6, img_h=96, img_w=96, spatial_mul=8)))
        roipool = 'roi_pool_{}'.format(idx)
        n[roipool] = L.ROIPooling(n.pool3, n[rois], roi_pooling_param=dict(pooled_w=7, pooled_h=7, spatial_scale=0.125))
        # fc
        fc1 = 'fc1_{}'.format(idx)
        relu6 = 'relu6_{}'.format(idx)
        drop1 = 'drop1_{}'.format(idx)
        n[fc1], n[relu6], n[drop1] = fc_relu_dropout(n[roipool], 2048, 0.5)
    # structure connection
    connect_structure_1 = [[0,2,4], [0,6,8], [0,10,12], [0,14,16], [0,18,20]]
    concate_bottom_final = []
    for idx in xrange(len(connect_structure_1)):
        concate_bottom = []
        for jdx in xrange(len(connect_structure_1[idx])):
            drop1 = 'drop1_{}'.format(connect_structure_1[idx][jdx])
            concate_bottom.append(n[drop1])
        concate_1 = 'concate_1_{}'.format(idx)
        n[concate_1] = L.Concat(*concate_bottom)
        fc2 = 'fc2_{}'.format(idx)
        relu7 = 'relu7_{}'.format(idx)
        drop2 = 'drop2_{}'.format(idx)
        n[fc2], n[relu7], n[drop2] = fc_relu_dropout(n[concate_1], 2048, 0.5)
        concate_bottom_final.append(n[drop2])

    n.fc_concat = L.Concat(*concate_bottom_final)
    n.fc3_0 = fc(n.fc_concat, point_num_ * 3)

    # loss
    if net_type == 'train':
        n.loss = L.SmoothL1Loss(n.fc3_0, n.pose,
                                smooth_l1_loss_param=dict(sigma=10),
                                loss_weight=1)
        n.distance = L.PoseDistance(n.fc3_0, n.pose, n.center, loss_weight=0,
                                    pose_distance_param=dict(cube_length=150, fx=fx_, fy=fy_, ux=ux_, uy=uy_),
                                    include=dict(phase=1))
        return first_layer + str(n.to_proto())
    else:
        n.error, n.output = L.PoseDistance(n.fc3_0, n.pose, n.center,
                                           pose_distance_param=dict(cube_length=150, fx=fx_, fy=fy_, ux=ux_, uy=uy_,
                                                                    output_pose=True),
                                           include=dict(phase=1), ntop=2)
        return str(n.to_proto())


def make_solver(iter_idx, output_dir, test_id):
    solver_content = 'net: \"{0}/model/train_msra_pose_ren_{2}_s{1}.prototxt\"\n\
test_iter: 67\n\
test_interval: 1000\n\
base_lr: 0.001\n\
lr_policy: \"step\"\n\
gamma: 0.1\n\
stepsize: 20000\n\
display: 100\n\
max_iter: 80000\n\
momentum: 0.9\n\
weight_decay: 0.0005\n\
snapshot: 40000\n\
snapshot_prefix: \"{0}/snapshot/msra_pose_ren_{2}_s{1}\"'.format(output_dir, iter_idx, test_id)
    return solver_content


def make_pose_ren_net(output_dir, test_id = 0, iter_num=2, test_iter_num=3):
    for iter_idx in xrange(1, iter_num + 1):
        with open('{}/model/train_msra_pose_ren_{}_s{}.prototxt'.format(output_dir, test_id, iter_idx), 'w') as f:
            f.write(pose_ren_net('train', iter_idx, output_dir, test_id))
        with open('{}/model/solver_msra_pose_ren_{}_s{}.prototxt'.format(output_dir, test_id, iter_idx), 'w') as f:
            f.write(make_solver(iter_idx, output_dir, test_id))
    for iter_idx in xrange(1, test_iter_num + 1):
        with open('{}/model/test_msra_pose_ren_{}_s{}_trainset.prototxt'.format(output_dir, test_id, iter_idx), 'w') as f:
            f.write(pose_ren_net('test-train', iter_idx, output_dir, test_id))
        with open('{}/model/test_msra_pose_ren_{}_s{}_testset.prototxt'.format(output_dir, test_id, iter_idx), 'w') as f:
            f.write(pose_ren_net('test-test', iter_idx, output_dir, test_id))


if __name__ == '__main__':
    make_pose_ren_net('.')
