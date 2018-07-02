import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add src to PYTHONPATH
sys.path.append(os.path.dirname(BASE_DIR))

# Add caffe to PYTHONPATH
sys.path.append(os.path.join(BASE_DIR, '../caffe-pose/python'))

# Add caffe_util to PYTHONPATH
sys.path.append(os.path.join(BASE_DIR, 'libs'))

# dataset dir: where to find train/test depth images
nyu_data_dir = '/home/workspace/Datasets/NYU/'
icvl_data_dir = '/home/workspace/Datasets/ICVL/'
msra_data_dir = '/home/workspace/Datasets/MSRA/cvpr15_MSRAHandGestureDB/'

# annotation dir: where to find train/test_image/label.txt
icvl_anno_dir = '../../caffe-pose/examples/hand_pose_estimation/annotations/icvl/'
nyu_anno_dir = '../../caffe-pose/examples/hand_pose_estimation/annotations/nyu/'
msra_anno_dir = '../../caffe-pose/examples/hand_pose_estimation/annotations/msra/'

# executable file: where to find output_pose, caffe
output_pose_bin_dir = '../../caffe-pose/build/tools/output_pose'
caffe_bin_dir = '../../caffe-pose/build/tools/caffe'

train_iter_num = 2
test_iter_num = 3