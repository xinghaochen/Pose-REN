import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
import util
import config
from model_pose_ren import ModelPoseREN
import numpy as np

from net_deploy_baseline import make_baseline_net
from net_deploy_pose_ren import make_pose_ren_net

def print_usage():
    print('usage: {} icvl/nyu model_prefix out_file base_dir'.format(sys.argv[0]))
    exit(-1)

def main():
    if len(sys.argv) < 3:
        print_usage()

    dataset = sys.argv[1]
    out_file = sys.argv[2]
    data_dir_dict = {'nyu': config.nyu_data_dir,
                     'icvl': config.icvl_data_dir + 'test/Depth/',
                     'msra': config.msra_data_dir}
    base_dir = data_dir_dict[dataset] #sys.argv[3]
    batch_size = 64
    if len(sys.argv) == 4:
        batch_size = int(sys.argv[3])

    # generate deploy prototxt
    make_baseline_net(os.path.join(ROOT_DIR, '../models'), dataset)
    make_pose_ren_net(os.path.join(ROOT_DIR, '../models'), dataset)

    hand_model = ModelPoseREN(dataset)
    names = util.load_names(dataset)
    centers = util.load_centers(dataset)
    if dataset == 'msra':
        # the last index of frames belong to the same subject
        msra_id_split_range = np.array([8499, 16991, 25403, 33891, 42391, 50888, 59385, 67883, 76375]) - 1
        results = []
        for test_id in xrange(9):
            hand_model.reset_model(dataset, test_id)
            sidx = msra_id_split_range[test_id-1] + 1 if test_id else 0
            eidx = msra_id_split_range[test_id]
            sub_names = names[sidx:eidx+1]
            sub_centers= centers[sidx:eidx + 1]
            print 'evaluating for subject {} ...'.format(test_id)
            sub_results = hand_model.detect_files(base_dir, sub_names, sub_centers, max_batch=batch_size)
            if test_id == 0:
                results = sub_results
            else:
                results = np.concatenate((results, sub_results), axis=0)
        util.save_results(results, out_file)
    else:
        results = hand_model.detect_files(base_dir, names, centers, max_batch=batch_size)
        util.save_results(results, out_file)

if __name__ == '__main__':
    main()
