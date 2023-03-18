import caffe
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'model_msra'))
from net_hand_baseline_wide_guided_structure_s1_msra import make_net

def combine_gt_init_label(gt_label_file, init_label_file, combined_label_file):
    J = 21
    # read lables
    with open(gt_label_file, 'r') as f:
        gt_labels = f.readlines()
    print len(gt_labels)
    with open(init_label_file, 'r') as f:
        init_labels = f.readlines()
    print len(init_labels)

    combined_poses = np.zeros((len(gt_labels), J*3*2), dtype=float)
    for idx in xrange(len(gt_labels)):
        # get pose
        gt_pose = map(float, gt_labels[idx].split())
        gt_pose = np.asarray(gt_pose)
        init_pose = map(float, init_labels[idx].split())
        init_pose = np.asarray(init_pose)
        combined_poses[idx] = np.concatenate((gt_pose, init_pose), axis=0)

    with open(combined_label_file, 'w') as f:
        for i in range(len(combined_poses)):
            for j in range(combined_poses[i].shape[0]):
                f.write('{:.3f} '.format(combined_poses[i][j]))
            f.write('\n')


def combine_two_files(file1, file2, combined_file):
    filenames = [file1, file2]
    with open(combined_file, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def make_output_pose_command(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy, test_id, iter):
    command = '{0} \
    --model={1} \
    --gpu=0 \
    --weights={2} \
    --label_list={3} \
    --output_name={4} \
    --fx={5} \
    --fy={6} \
    --ux={7} \
    --uy={8} \
    2>&1 | tee test_msra_log_{9}_iter{10}.txt'.format(output_pose_bin, model, weights, label_list, output_name, fx, fy, ux, uy, test_id, iter)
    return command

# make net
make_net()

# init caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# parameters
root_dir = '/home/workspace/Datasets/MSRA/cvpr15_MSRAHandGestureDB/'
output_pose_bin = '/home/chenxh/cxh/Projects/Handpose/cnn-pose/build/tools/output_pose'
test_id = sys.argv[1]
fx = 241.42
fy = 241.42
ux = 160
uy = 120

# --------------------------------------------------------------------------
# stage 1
# --------------------------------------------------------------------------
iter_num = 2
for iter in xrange(iter_num+1):
    print 'start iter {} ...'.format(iter)
    # prepare input files
    gt_label_file = '{}/test_label_{}.txt'.format(root_dir, test_id)
    # test label
    if iter == 0:
        # get test label from model0
        model = 'model_msra/test_hand_baseline_wide_{}.prototxt'.format(test_id)
        weights = 'snapshot_msra/hand_baseline_wide_{}_iter_80000.caffemodel'.format(test_id)
        label_list = root_dir + 'test_label_{}.txt'.format(test_id)
        init_label_file_stage0 = 'output/test_msra_hand_baseline_wide_{}.txt'.format(test_id)
        cmd = make_output_pose_command(output_pose_bin, model, weights, label_list, init_label_file_stage0, fx, fy, ux, uy, test_id, iter)
        os.system(cmd)
        # combine groundtruth label and init label
        combined_label_file_stage0 = 'model_msra/init/test_combined_label_{}.txt'.format(test_id)
        gt_label_file = label_list
        combine_gt_init_label(gt_label_file, init_label_file_stage0, combined_label_file_stage0)
        print 'finish iter {} ...'.format(iter)
        continue

    # get train label from model{iter-1}
    if iter == 1:
        model = 'model_msra/test_hand_baseline_wide_{}_train.prototxt'.format(test_id)
        weights = 'snapshot_msra/hand_baseline_wide_{}_iter_80000.caffemodel'.format(test_id)
        init_label_file_train = 'output/train_msra_hand_baseline_wide_{}.txt'.format(test_id)
    else:
        model = 'model_msra/test_hand_baseline_wide_guided_structure_s1_{}_stage{}.prototxt'.format(test_id, iter-1)
        weights = 'snapshot_msra/hand_baseline_wide_guided_structure_s1_{}_stage{}_iter_80000.caffemodel'.format(test_id, iter-1)
        # label_list = 'model_msra/init/train_combined_label_{}_stage{}.txt'.format(test_id, iter-1)
        init_label_file_train = 'output/train_msra_hand_baseline_wide_guided_structure_s1_{}_stage{}.txt'.format(test_id, iter-1)
    label_list = root_dir + 'train_label_{}.txt'.format(test_id)
    cmd = make_output_pose_command(output_pose_bin, model, weights, label_list, init_label_file_train, fx, fy, ux, uy, test_id, iter)
    os.system(cmd)
    # combine groundtruth label and init label
    if iter > 1:
        combined_label_file_stage = 'model_msra/init/train_combined_label_{}_stage{}_single.txt'.format(test_id, iter)
    else:
        combined_label_file_stage = 'model_msra/init/train_combined_label_{}_stage{}.txt'.format(test_id, iter)
    gt_label_file = label_list
    combine_gt_init_label(gt_label_file, init_label_file_train, combined_label_file_stage)
    # combine all training samples from model{1:iter}
    if iter > 1:
        file_prev = 'model_msra/init/train_combined_label_{}_stage{}.txt'.format(test_id, iter-1)
        combined_file = 'model_msra/init/train_combined_label_{}_stage{}.txt'.format(test_id, iter)
        combine_two_files(file_prev, combined_label_file_stage, combined_file)
        image_list = root_dir + 'train_image_{}.txt'.format(test_id)
        combined_file = 'model_msra/init/train_double_image_{}.txt'.format(test_id)
        combine_two_files(image_list, image_list, combined_file)

    # solve
    print 'start solving iter {} ...'.format(iter)
    solver_name = 'model_msra/solver_hand_baseline_wide_guided_structure_s1_{}_stage{}.prototxt'.format(test_id, iter)
    solver = caffe.SGDSolver(solver_name)
    if iter == 1:
        solver.net.copy_from('snapshot_msra/hand_baseline_wide_{}_iter_80000.caffemodel'.format(test_id))
    elif iter > 1:
        solver.net.copy_from('snapshot_msra/hand_baseline_wide_guided_structure_s1_{0}_stage{1}_iter_80000.caffemodel'.format(test_id, iter-1))
    # solver.restore('snapshot/train_iter_70000.solverstate')
    solver.solve()
    print 'finish solving iter {} ...'.format(iter)
    print 'finish iter {} ...'.format(iter)
