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


def deploy_baseline_net(dataset):
    n = caffe.NetSpec()
    point_num_ = util.get_joint_num(dataset)
    # input layers
    # n.data = 'input: \"data\"\
    #                 input_shape \{\
    #                     dim: 1\
    #                     dim: 1\
    #                     dim: 96\
    #                     dim: 96\
    #                 \}'
    # n.prev_pose = 'input: \"prev_pose\"\
    #                 input_shape \{\
    #                     dim: 1\
    #                     dim: {}\
    #                 \}'.format(point_num_*3)
    n.data = L.Input(name="data", shape=dict(dim=[1, 1, 96, 96]))
    n.prev_pose = L.Input(name="prev_pose", shape=dict(dim=[1, point_num_*3]))

    print str(n.to_proto())

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

    return str(n.to_proto())

def make_baseline_net(output_dir, dataset):
    with open('{}/deploy_{}_baseline.prototxt'.format(output_dir, dataset), 'w') as f:
        f.write(deploy_baseline_net(dataset))

if __name__ == '__main__':
    make_baseline_net('../../models', 'nyu')
