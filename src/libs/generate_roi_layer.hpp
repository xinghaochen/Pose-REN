#ifndef CAFFE_GENERATE_ROI_LAYER_HPP_
#define CAFFE_GENERATE_ROI_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GenerateROILayer : public Layer<Dtype> {
public:
	explicit GenerateROILayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "GenerateROI"; }

	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MaxBottomBlobs() const { return 2; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int joint_idx;
	int roi_h;
	int roi_w;
	int img_h;
	int img_w;
	Dtype spatial_mul;
			
	int channels_;
	int height_;
	int width_;
};

}  // namespace caffe

#endif  // CAFFE_GENERATE_ROI_LAYER_HPP_
