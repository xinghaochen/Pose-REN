// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/generate_roi_layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

#if _MSC_VER < 1800
inline double round(double x) {
	return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}
#endif

namespace caffe {

	template <typename Dtype>
	void GenerateROILayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			GenerateROIParameter generate_roi_param = this->layer_param_.generate_roi_param();
			joint_idx = generate_roi_param.joint_idx();
			roi_h = generate_roi_param.roi_h();
			roi_w = generate_roi_param.roi_w();
			img_h = generate_roi_param.img_h();
			img_w = generate_roi_param.img_w();
			spatial_mul = generate_roi_param.spatial_mul();
		
			CHECK_GE(joint_idx, 0)
				<< "joint_idx must be >= 0";
	}

	template <typename Dtype>
	void GenerateROILayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			channels_ = bottom[0]->channels();
			height_ = bottom[0]->height();
			width_ = bottom[0]->width();
			// c++ 11
			// vector<int> s = {bottom[0]->num(), 5};
			// c+ 11
			vector<int> s;
			s.push_back(bottom[0]->num());
			s.push_back(5);
			top[0]->Reshape(s);
	}

	template <typename Dtype>
	void GenerateROILayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			const Dtype* poses = bottom[0]->cpu_data();
			int batch_size = bottom[0]->num();
			int channel_num = bottom[0]->channels();
			int top_count = top[0]->count();
			int pose_dim = bottom[0]->count()/batch_size;
			Dtype* top_data = top[0]->mutable_cpu_data();
			caffe_set(top_count, Dtype(0), top_data);
			
			//std::cout << "1: " << batch_size << " " << channel_num << " " << top_count << " " << pose_dim << std::endl;
			//std::cout << "1.1: " << roi_w << " " << roi_h << " " << img_h << " " << spatial_mul << std::endl;
			for (int n = 0; n < batch_size; ++n) {
				Dtype x1 = (poses[n*pose_dim + joint_idx*3 + 0]+1)*img_w/2 - roi_w*spatial_mul/2;
				Dtype y1 = (poses[n*pose_dim + joint_idx*3 + 1]+1)*img_h/2 - roi_h*spatial_mul/2;
				Dtype x2 = (poses[n*pose_dim + joint_idx*3 + 0]+1)*img_w/2 + roi_w*spatial_mul/2;
				Dtype y2 = (poses[n*pose_dim + joint_idx*3 + 1]+1)*img_h/2 + roi_h*spatial_mul/2;
				//std::cout << "2: " << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
				if (x1 < 0) {
					x1 = 0;
					x2 = x1 + roi_w*spatial_mul;
				}
				if (x2 >= img_w-1) {
					x2 = img_w-1;
					x1 = x2 - roi_w*spatial_mul;
				}
				if (y1 < 0) {
					y1 = 0;
					y2 = y1 + roi_h*spatial_mul;
				}
				if (y2 >= img_h-1) {
					y2 = img_h-1;
					y1 = y2 - roi_h*spatial_mul;
				}
				//std::cout << "3: " << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
				top_data[n*5 + 0] = n; //[i, x1, y1, x2, y2]
				top_data[n*5 + 1] = x1;
				top_data[n*5 + 2] = y1;
				top_data[n*5 + 3] = x2;
				top_data[n*5 + 4] = y2;
			}
	}
	
	template <typename Dtype>
	void GenerateROILayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		  NOT_IMPLEMENTED;
    }



    INSTANTIATE_CLASS(GenerateROILayer);
    REGISTER_LAYER_CLASS(GenerateROI);

    }  // namespace caffe
