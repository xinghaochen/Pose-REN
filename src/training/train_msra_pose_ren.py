import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)  # config
sys.path.append(os.path.join(ROOT_DIR, 'utils'))  # utils

import config
import caffe
import numpy as np
from net_msra_pose_ren import make_pose_ren_net
from net_msra_baseline import make_baseline_net

from pose_ren_util import *
import util

# print sys.path

# init caffe
if len(sys.argv) >= 3:
    gpu_id = int(sys.argv[2])
    test_id = int(sys.argv[1])
elif len(sys.argv) >= 2:
    test_id = int(sys.argv[1])
    gpu_id = 0
else:
    gpu_id = 0
    test_id = 0

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

# parameters
root_dir = config.msra_data_dir
anno_dir = config.msra_anno_dir
output_pose_bin = config.output_pose_bin_dir
dataset = 'msra'
fx, fy, ux, uy = util.get_param(dataset)
train_iter_num = config.train_iter_num
test_iter_num = config.test_iter_num
point_num = util.get_joint_num(dataset)

if not os.path.exists(output_pose_bin):
    print 'File not exist: {}'.format(output_pose_bin)
    exit()

# prepare folder
output_dir = '../../output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join('../../output', dataset)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir = os.path.join(output_dir, 'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
snapshot_dir = os.path.join(output_dir, 'snapshot')
if not os.path.exists(snapshot_dir):
    os.mkdir(snapshot_dir)
cache_dir = os.path.join(output_dir, 'cache')
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
log_dir = os.path.join(output_dir, 'logs')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# generate model prototxt
make_pose_ren_net(output_dir, test_id, train_iter_num, test_iter_num)
make_baseline_net(output_dir, test_id)

# models and weights
models_test = ['{}/test_msra_baseline_{}_testset.prototxt'.format(model_dir, test_id)]  # init model: baseline model
for idx in xrange(1, test_iter_num):
    models_test.append('{}/test_msra_pose_ren_{}_s{}_testset.prototxt'.format(model_dir, test_id, idx))

models_train = ['{}/test_msra_baseline_{}_trainset.prototxt'.format(model_dir, test_id)]  # init model: baseline model
for idx in xrange(1, test_iter_num):
    models_train.append('{}/test_msra_pose_ren_{}_s{}_trainset.prototxt'.format(model_dir, test_id, idx))

weights = ['../../models/msra_baseline_{}.caffemodel'.format(test_id)]  # init model: baseline model
for idx in xrange(1, train_iter_num):
    weights.append('{}/msra_pose_ren_{}_s{}_iter_80000.caffemodel'.format(snapshot_dir, test_id, idx))

# prepare input files
gt_label_test = '{}/test_label_{}.txt'.format(anno_dir, test_id)
gt_label_train = '{}/train_label_{}.txt'.format(anno_dir, test_id)
image_train = '{}/train_image_{}.txt'.format(anno_dir, test_id)
image_test = '{}/test_image_{}.txt'.format(anno_dir, test_id)
# copy files to cache
os.system('cp {} {}'.format(gt_label_test, cache_dir))
os.system('cp {} {}'.format(gt_label_train, cache_dir))
os.system('cp {} {}'.format(image_test, cache_dir))
os.system('cp {} {}'.format(image_train, cache_dir))

# --------------------------------------------------------------------------
# start training Pose-REN
# --------------------------------------------------------------------------
for iter_idx in xrange(1, train_iter_num + 1):
    print 'start iter_idx {} ...'.format(iter_idx)

    # get test label from  from model{iter_idx-1}
    if not os.path.exists(weights[iter_idx - 1]):
        print 'File not exist: {}'.format(weights[iter_idx - 1])
        exit()
    init_label_test = '{}/predicted_{}_s{}_testset.txt'.format(cache_dir, test_id, iter_idx - 1)
    log_suffix = '{}/log_test_msra_{}_testset'.format(log_dir, test_id)
    cmd = make_output_pose_command(output_pose_bin, models_test[iter_idx - 1], weights[iter_idx - 1], gt_label_test,
                                   init_label_test, fx, fy, ux, uy, iter_idx, gpu_id, log_suffix)
    os.system(cmd)
    # combine groundtruth label and init label
    test_label_iter = '{}/test_label_{}_s{}.txt'.format(cache_dir, test_id, iter_idx)
    combine_gt_init_label(gt_label_test, init_label_test, test_label_iter, J = point_num)

    # get train label from model{iter_idx-1}
    prev_label_train = '{}/predicted_{}_s{}_trainset.txt'.format(cache_dir, test_id, iter_idx - 1)
    log_suffix = '{}/log_test_msra_{}_trainset'.format(log_dir, test_id)
    cmd = make_output_pose_command(output_pose_bin, models_train[iter_idx - 1], weights[iter_idx - 1], gt_label_train,
                                   prev_label_train, fx, fy, ux, uy, iter_idx, gpu_id, log_suffix)
    os.system(cmd)
    # combine groundtruth label and init label
    combined_label_train = '{}/train_label_{}_s{}_single.txt'.format(cache_dir, test_id, iter_idx)
    combine_gt_init_label(gt_label_train, prev_label_train, combined_label_train, J = point_num)
    print 'saving {} ...'.format(combined_label_train)

    # combine all training samples from model{1:iter_idx}
    if iter_idx == 1:
        train_label_iter = '{}/train_label_{}_s{}.txt'.format(cache_dir, test_id, iter_idx)
        train_image_iter = '{}/train_image_{}_s{}.txt'.format(cache_dir, test_id, iter_idx)
        os.system('cp {} {}'.format(combined_label_train, train_label_iter))
        os.system('cp {} {}'.format(image_train, train_image_iter))
    elif iter_idx > 1:
        combined_label_train_list = []
        combined_image_train_list = []
        for idx in xrange(1, iter_idx + 1):
            prev_label_train = '{}/train_label_{}_s{}_single.txt'.format(cache_dir, test_id, iter_idx - 1)
            combined_label_train_list.append(prev_label_train)
            combined_image_train_list.append(image_train)
        train_label_iter = '{}/train_label_{}_s{}.txt'.format(cache_dir, test_id, iter_idx)
        combine_files(combined_label_train_list, train_label_iter)
        train_image_iter = '{}/train_image_{}_s{}.txt'.format(cache_dir, test_id, iter_idx)
        combine_files(combined_image_train_list, train_image_iter, True)

    # solve
    print 'start solving iter_idx {} ...'.format(iter_idx)
    solver_name = '{}/solver_msra_pose_ren_{}_s{}.prototxt'.format(model_dir, test_id, iter_idx)
    solver = caffe.SGDSolver(solver_name)
    solver.net.copy_from(weights[iter_idx - 1])
    solver.solve()
    print 'finish solving iter_idx {} ...'.format(iter_idx)
    print 'finish iter_idx {} ...'.format(iter_idx)
