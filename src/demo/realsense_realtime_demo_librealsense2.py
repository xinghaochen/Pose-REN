import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import pyrealsense2 as rs
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
from model_pose_ren import ModelPoseREN
import util
from util import get_center_fast as get_center

def init_device():
    # Configure depth streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    print 'config'
    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print "Depth Scale is: " , depth_scale
    return pipeline, depth_scale

def stop_device(pipeline):
    pipeline.stop()
    
def read_frame_from_device(pipeline, depth_scale):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    #if not depth_frame:
    #    return None
    # Convert images to numpy arrays
    depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)
    depth = depth_image * depth_scale * 1000
    return depth

def show_results(img, results, cropped_image, dataset):
    img = np.minimum(img, 1500)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img*255)
    # draw cropped image
    img[:96, :96] = (cropped_image+1)*255/2
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (0, 0), (96, 96), (255, 0, 0))
    img_show = util.draw_pose(dataset, img, results)
    return img_show

def main():
    # intrinsic paramters of Intel Realsense SR300
    fx, fy, ux, uy = 463.889, 463.889, 320, 240
    # paramters
    dataset = 'icvl'
    if len(sys.argv) == 2:
        dataset = sys.argv[1]

    lower_ = 1
    upper_ = 650

    # init realsense
    pipeline, depth_scale = init_device()
    # init hand pose estimation model
    hand_model = ModelPoseREN(dataset,
        lambda img: get_center(img, lower=lower_, upper=upper_),
        param=(fx, fy, ux, uy), use_gpu=True)
    # for msra dataset, use the weights for first split
    if dataset == 'msra':
        hand_model.reset_model(dataset, test_id = 0)
    # realtime hand pose estimation loop
    while True:
        depth = read_frame_from_device(pipeline, depth_scale)
        # preprocessing depth
        depth[depth == 0] = depth.max()
        # training samples are left hands in icvl dataset,
        # right hands in nyu dataset and msra dataset,
        # for this demo you should use your right hand
        if dataset == 'icvl':
            depth = depth[:, ::-1]  # flip
        # get hand pose
        results, cropped_image = hand_model.detect_image(depth)
        img_show = show_results(depth, results, cropped_image, dataset)
        cv2.imshow('result', img_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stop_device(pipeline)

if __name__ == '__main__':
    main()
