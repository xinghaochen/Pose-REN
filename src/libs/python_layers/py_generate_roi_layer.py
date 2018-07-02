
import caffe
import numpy as np

class PyGenerateROILayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input
        if len(bottom) != 1:
            raise Exception("PyGenerateROILayer only takes pose as input.")
        self.batch_size = bottom[0].data.shape[0]
        self.joint_num = int(np.floor(bottom[0].data.shape[1] / 3))
        # config
        params = eval(self.param_str)
        self.joint_idx = int(params['joint_idx'])
        self.roi_h = params['roi_h']
        self.roi_w = params['roi_w']
        self.img_h = params['img_h']
        self.img_w = params['img_w']
        self.spatial_mul = params['spatial_mul']
        if self.joint_idx < 0 or self.joint_idx >= self.joint_num:
            raise Exception("PyGenerateROILayer only takes pose as input.")

    def reshape(self, bottom, top):
        # rois: (batch_ind, x1, y1, x2, y2)
        self.batch_size = bottom[0].data.shape[0]
        top[0].reshape(self.batch_size, 5)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
    
    def forward(self, bottom, top):
        poses = np.array(bottom[0].data)
        top[0].data[...] = top[0].data[...] * 0
        for i in xrange(self.batch_size):
            joint = poses[i]
            joint = np.reshape(joint, (-1, 3))
            x1 = (joint[self.joint_idx, 0]+1)*self.img_w/2 - self.roi_w*self.spatial_mul/2
            y1 = (joint[self.joint_idx, 1]+1)*self.img_h/2 - self.roi_h*self.spatial_mul/2
            x2 = (joint[self.joint_idx, 0]+1)*self.img_w/2 + self.roi_w*self.spatial_mul/2
            y2 = (joint[self.joint_idx, 1]+1)*self.img_h/2 + self.roi_h*self.spatial_mul/2
            if x1 < 0:
                x1 = 0
                x2 = x1 + self.roi_w*self.spatial_mul
            if x2 >= self.img_w-1:
                x2 = self.img_w-1
                x1 = x2 - self.roi_w*self.spatial_mul
            if y1 < 0:
                y1 = 0
                y2 = y1 + self.roi_h*self.spatial_mul
            if y2 >= self.img_h-1:
                y2 = self.img_h-1
                y1 = y2 - self.roi_h*self.spatial_mul
            top[0].data[i, :] = [i, x1, y1, x2, y2]

    def backward(self, top, propatate_down, bottom):
        pass
