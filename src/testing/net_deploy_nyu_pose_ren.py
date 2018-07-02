import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
import config
import caffe
from caffe import layers as L, params as P
import util
from caffe_util import *

def deploy_pose_ren_net():
    dataset = 'nyu'
    n = caffe.NetSpec()
    point_num_ = util.get_joint_num(dataset)

    n.data = L.Input(name="data", shape=dict(dim=[1, 1, 96, 96]))
    n.prev_pose = L.Input(name="prev_pose", shape=dict(dim=[1, point_num_*3]))

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
        if idx == 1 or idx == 2 or idx == 4:
            continue
        rois = 'rois_{}'.format(idx)
        n[rois] = L.Python(n.prev_pose, module='layers.py_generate_roi_layer',
            layer='GenerateROILayer', ntop=1,
            param_str=str(dict(joint_idx=idx, roi_h=6, roi_w=6, img_h=96, img_w=96, spatial_mul=8)))
        roipool = 'roi_pool_{}'.format(idx)
        n[roipool] = L.ROIPooling(n.pool3, n[rois], roi_pooling_param=dict(pooled_w=7, pooled_h=7, spatial_scale=0.125))
        # fc
        fc1 = 'fc1_{}'.format(idx)
        relu6 = 'relu6_{}'.format(idx)
        drop1 = 'drop1_{}'.format(idx)
        n[fc1], n[relu6], n[drop1] = fc_relu_dropout(n[roipool], 2048, 0.5)
    # structure connection
    connect_structure_1 = [[0,5,3], [0,13,12], [0,11,10], [0,9,8], [0,7,6]]
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
    n.fc3_0 = fc(n.fc_concat, point_num_*3)

    return str(n.to_proto())


def make_pose_ren_net(output_dir):
    with open('{}/deploy_nyu_pose_ren.prototxt'.format(output_dir), 'w') as f:
        f.write(deploy_pose_ren_net())


if __name__ == '__main__':
    make_pose_ren_net('../../models')