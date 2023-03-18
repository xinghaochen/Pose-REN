import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)  # config
sys.path.append(os.path.join(ROOT_DIR, 'utils'))  # utils
import config
import caffe
import numpy as np
from net_nyu_pose_ren import make_pose_ren_net
from net_nyu_baseline import make_baseline_net

from pose_ren_util import *
import util

# init caffe
if len(sys.argv) >= 2:
    gpu_id = int(sys.argv[1])
else:
    gpu_id = 0
caffe.set_device(gpu_id)
caffe.set_mode_gpu()

# parameters
root_dir = config.nyu_data_dir
anno_dir = config.nyu_anno_dir
output_pose_bin = config.output_pose_bin_dir
dataset = 'nyu'
fx, fy, ux, uy = util.get_param(dataset)
train_iter_num = config.train_iter_num
test_iter_num = config.test_iter_num
point_num = util.get_joint_num(dataset)

if not os.path.exists(output_pose_bin):
    print 'File not exist: {}'.format(output_pose_bin)
    exit()

# prepare folder
output_dir = os.path.join('../../output', dataset)
model_dir = os.path.join(output_dir, 'model')
snapshot_dir = os.path.join(output_dir, 'snapshot')
cache_dir = os.path.join(output_dir, 'cache')
log_dir = os.path.join(output_dir, 'logs')
result_dir = os.path.join(output_dir, 'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# generate model prototxt
make_pose_ren_net(output_dir, train_iter_num, test_iter_num)
make_baseline_net(output_dir)

# models and weights
models_test = ['{}/test_nyu_baseline_testset.prototxt'.format(model_dir)] # init model: baseline model
for idx in xrange(1, test_iter_num+1):
    models_test.append('{}/test_nyu_pose_ren_s{}_testset.prototxt'.format(model_dir, idx))

models_train = ['{}/test_nyu_baseline_trainset.prototxt'.format(model_dir)] # init model: baseline model
for idx in xrange(1, train_iter_num+1):
    models_train.append('{}/test_nyu_pose_ren_s{}_trainset.prototxt'.format(model_dir, idx))

weights_init = '../../models/nyu_baseline.caffemodel'.format(snapshot_dir) # init model: baseline model
weights_pose_ren = '{}/nyu_pose_ren_s{}_iter_160000.caffemodel'.format(snapshot_dir, idx)

# prepare input files
gt_label_test = '{}/test_label.txt'.format(root_dir)

# --------------------------------------------------------------------------
# start testing Pose-REN
# --------------------------------------------------------------------------
iter_num = 3
for iter_idx in xrange(iter_num+1):
    print 'start iter_idx {} ...'.format(iter_idx)

    weights = weights_pose_ren if iter_idx else weights_init
    # get test label from model{iter_idx-1}
    if not os.path.exists(weights):
        print 'File not exist: {}'.format(weights)
        exit()
    init_label_test = '{}/predicted_nyu_pose_ren_s{}.txt'.format(result_dir, iter_idx)
    log_suffix = '{}/log_final_test_nyu'.format(log_dir)
    cmd = make_output_pose_command(output_pose_bin, models_test[iter_idx], weights, gt_label_test,
                                   init_label_test, fx, fy, ux, uy, iter_idx, gpu_id, log_suffix)
    os.system(cmd)
    if iter_idx <= iter_num:
        # combine groundtruth label and init label
        combined_label = '{}/test_label_s{}.txt'.format(cache_dir, iter_idx+1)
        combine_gt_init_label(gt_label_test, init_label_test, combined_label, J = point_num)
    print 'finish iter_idx {} ...'.format(iter_idx)
