import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR)) # config
import config
import caffe
from caffe import layers as L, params as P
import util
from caffe_util import *

    
def hand_baseline(net_type, output_dir, test_id = 0):
    dataset = 'msra'
    n = caffe.NetSpec()
    fx_, fy_, ux_, uy_ = util.get_param(dataset)
    point_num_ = util.get_joint_num(dataset)
    root_folder_ = config.msra_data_dir

    # data layers
    if net_type == 'train':
        image_source_=root_folder_+'train_image_{}.txt'.format(test_id)
        pose_data_param_train = dict(image_source=image_source_,
                                     label_source=root_folder_+'train_label_{}.txt'.format(test_id),
                                     root_folder=root_folder_,
                                     batch_size=128, shuffle=True, new_height=96, new_width=96,
                                     point_num=point_num_, point_dim=3,
                                     cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.MSRA)
        n.data, n.pose = L.PoseData(name="data", include=dict(phase=0),
                     transform_param=dict(is_trans=True, trans_dx=10, trans_dy=10, is_rotate=True, rotate_deg=180, is_zoom=True, zoom_scale=0.1),
                     pose_data_param=pose_data_param_train, ntop=2)
        first_layer = str(n.to_proto())
        
        pose_data_param_test = dict(image_source=root_folder_ +'test_image_{}.txt'.format(test_id),
                                 label_source=root_folder_+'test_label_{}.txt'.format(test_id),
                                 root_folder=root_folder_,
                                 batch_size=128, shuffle=False, new_height=96, new_width=96,
                                 point_num=point_num_, point_dim=3, output_center=True,
                                 cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.MSRA)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=1),
                     transform_param=dict(is_trans=False, is_rotate=False, is_zoom=False),
                     pose_data_param=pose_data_param_test, ntop=2)
        n.pose, n.center = L.Slice(n.label, slice_param=dict(slice_dim=1, slice_point=point_num_*3), include=dict(phase=1), ntop=2)
    elif net_type == 'test-train':
        pose_data_param_test = dict(image_source='{}/cache/train_image_{}.txt'.format(output_dir, test_id),
                                 label_source='{}/cache/train_label_{}.txt'.format(output_dir, test_id),
                                 root_folder=root_folder_,
                                 batch_size=128, shuffle=False, new_height=96, new_width=96,
                                 point_num=point_num_, point_dim=3, output_center=True,
                                 cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.MSRA)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=1),
                     transform_param=dict(is_trans=False, is_rotate=False, is_zoom=False),
                     pose_data_param=pose_data_param_test, ntop=2)
        n.pose, n.center = L.Slice(n.label, slice_param=dict(slice_dim=1, slice_point=point_num_*3), include=dict(phase=1), ntop=2)
    elif net_type == 'test-test':
        pose_data_param_test = dict(image_source=root_folder_ +'test_image_{}.txt'.format(test_id),
                                 label_source=root_folder_+'test_label_{}.txt'.format(test_id),
                                 root_folder=root_folder_,
                                 batch_size=128, shuffle=False, new_height=96, new_width=96,
                                 point_num=point_num_, point_dim=3, output_center=True,
                                 cube_length=150, fx=fx_, fy=fy_, dataset=P.PoseData.MSRA)
        n.data, n.label = L.PoseData(name="data", include=dict(phase=1),
                     transform_param=dict(is_trans=False, is_rotate=False, is_zoom=False),
                     pose_data_param=pose_data_param_test, ntop=2)
        n.pose, n.center = L.Slice(n.label, slice_param=dict(slice_dim=1, slice_point=point_num_*3), include=dict(phase=1), ntop=2)

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

    # fc
    n.fc1, n.relu6_0, n.drop1_0 = fc_relu_dropout(n.relu5, 2048, 0.5)
    n.fc2, n.relu7_0, n.drop2_0 = fc_relu_dropout(n.drop1_0, 2048, 0.5)
    n.fc3 = fc(n.drop2_0, point_num_*3)


    # loss
    if net_type == 'train':
        n.loss = L.SmoothL1Loss(n.fc3, n.pose,
                smooth_l1_loss_param=dict(sigma=10),
                loss_weight=1)
        n.distance = L.PoseDistance(n.fc3, n.pose, n.center, loss_weight=0,
                                    pose_distance_param=dict(cube_length=150, fx=fx_, fy=fy_, ux=ux_, uy=uy_),
                                    include=dict(phase=1))
        return first_layer + str(n.to_proto())
    else:
        n.error, n.output = L.PoseDistance(n.fc3, n.pose, n.center,
                                    pose_distance_param=dict(cube_length=150, fx=fx_, fy=fy_, ux=ux_, uy=uy_, output_pose=True),
                                    include=dict(phase=1), ntop=2)
        return str(n.to_proto())

def make_solver(test_id = 0):
    solver_content = 'net: \"model_msra/handpose_baseline_msra_{0}.prototxt\"\n\
test_iter: 12\n\
test_interval: 1000\n\
base_lr: 0.001\n\
lr_policy: \"step\"\n\
gamma: 0.1\n\
stepsize: 40000\n\
display: 100\n\
max_iter: 160000\n\
momentum: 0.9\n\
weight_decay: 0.0005\n\
snapshot: 160000\n\
snapshot_prefix: \"snapshot_msra/handpose_baseline_msra_{0}\"'.format(test_id)
    return solver_content

def make_baseline_net(output_dir, test_id = 0):
    # with open('model_msra/handpose_baseline_msra.prototxt'.format(iter), 'w') as f:
    #     f.write(hand_baseline('train', iter))
    with open('{}/model/test_msra_baseline_{}_testset.prototxt'.format(output_dir, test_id), 'w') as f:
        f.write(hand_baseline('test-test', output_dir, test_id))
    with open('{}/model/test_msra_baseline_{}_trainset.prototxt'.format(output_dir, test_id), 'w') as f:
        f.write(hand_baseline('test-train', output_dir, test_id))
    # with open('model_msra/solver_handpose_baseline_msra.prototxt'.format(iter), 'w') as f:
    #     f.write(make_solver(iter))

if __name__ == '__main__':
    make_baseline_net('.', 0)
