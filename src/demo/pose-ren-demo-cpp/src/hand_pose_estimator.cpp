
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

//#include <gflags/gflags.h>
//#include <glog/logging.h>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include "hand_pose_estimator.h"
#include <ctime>



using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
using std::vector;

HandPoseEstimator::HandPoseEstimator(string dataset) {
	gpu = 0;
	fx = 463.889;
	fy = 463.889;
	ux = 320;
	uy = 240;

	/*
	// test icvl
	fx = 240.99;
	fy = 240.96;
	ux = 160;
	uy = 120;
	*/

	lower = 10;
	upper = 650;
	cube_length.push_back(150);
	cube_length.push_back(150);
	cube_length.push_back(150);
	height = 96;
	width = 96;
	output_blob = "predict";
	output_blob_init = "predict";
	if (dataset.compare("hands17") == 0) {
        weights = "../../../models/model_baseline_hands17_full_large_aug.caffemodel";
        model = "../../../models/deploy_hands17_baseline.prototxt";
        weights_guided = "../../../models/model_pose_ren_hands17_full_large_aug.caffemodel";
        model_guided = "../../../models/deploy_hands17_pose_ren_py.prototxt";
	}
	else if (dataset.compare("nyu") == 0 || dataset.compare("icvl") == 0) {
        weights = "../../../models/" + dataset + "_baseline.caffemodel";
        model = "../../../models/deploy_" + dataset + "_baseline.prototxt";
        weights_guided = "../../../models/" + dataset + "_pose_ren.caffemodel";
        model_guided = "../../../models/deploy_" + dataset + "_pose_ren.prototxt";
        output_blob = "fc3_0";
	    output_blob_init = "fc3";
	}
	else if (dataset.compare("msra") == 0) {
        weights = "../../../models/" + dataset + "_baseline_0.caffemodel";
        model = "../../../models/deploy_" + dataset + "_baseline.prototxt";
        weights_guided = "../../../models/" + dataset + "_pose_ren_0.caffemodel";
        model_guided = "../../../models/deploy_" + dataset + "_pose_ren.prototxt";
        output_blob = "fc3_0";
	    output_blob_init = "fc3";
	}

	init_model();
}

HandPoseEstimator::~HandPoseEstimator() {

}

int HandPoseEstimator::init_model() {
	// initialize model
	if (gpu < 0) {
		LOG(INFO) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}
	else {
		LOG(INFO) << "Using GPU " << gpu;
		Caffe::SetDevice(gpu);
		Caffe::set_mode(Caffe::GPU);
	}
	test_net = boost::shared_ptr<Net<float> >(new Net<float>(model, caffe::TEST));
	test_net->CopyTrainedLayersFrom(weights);
	test_net_guided = boost::shared_ptr<Net<float> >(new Net<float>(model_guided, caffe::TEST));
	test_net_guided->CopyTrainedLayersFrom(weights_guided);
	return 1;
}

vector<float> HandPoseEstimator::predict_guided(const cv::Mat& cv_img, cv::Mat& crop, bool is_crop) {

	// get outputs and save
	const boost::shared_ptr<Blob<float> > blob =
		test_net->blob_by_name(output_blob_init);
	const boost::shared_ptr<Blob<float> > blob_guided =
		test_net_guided->blob_by_name(output_blob);
	int batch_size = blob->num();
	int res_size = blob->count() / batch_size;
	// pre processing
	vector<float> center;
	//cv::imshow("cv_img", cv_img);

	std::clock_t start;
	double duration;
	start = std::clock();

	if (is_crop) {
		crop = cv_img / (255.0 / 2) - 1;
	}
	else {
		get_center(cv_img, center, lower, upper);
		crop = crop_image(cv_img, center, cube_length, fx, fy, height, width);
		//cv::imshow("crop", crop);
	}
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "get center and crop image time: " << duration << '\n';

	//feed data
	boost::shared_ptr<Blob<float> > blob_data = test_net->blob_by_name("data");
	Mat2Blob(crop, blob_data);

	// forward
	start = std::clock();
	test_net->Forward();
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "baseline forward time: " << duration << '\n';
	const float* data = blob->cpu_data();

	// guided net
	start = std::clock();
	boost::shared_ptr<Blob<float> > blob_data_guided = test_net_guided->blob_by_name("data");
	Mat2Blob(crop, blob_data_guided);
	boost::shared_ptr<Blob<float> > blob_pose_guided = test_net_guided->blob_by_name("prev_pose");

    const float* prev_data = data;
    for (int iter=0; iter<2; iter++) {
	    Pose2Blob(prev_data, res_size, blob_pose_guided);
	    test_net_guided->Forward();
	    prev_data = blob_guided->cpu_data();
    }
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "pose ren iter twice feed data and forward time: " << duration << '\n';
	const float* data_guided = blob_guided->cpu_data();

	// debug output
	/*
	boost::shared_ptr<Blob<float> > roi = test_net_guided->blob_by_name("rois_0");
	const float* data_roi = roi->cpu_data();
	for (int k=0; k < 5; k++) {
		std::cout << *data_roi << std::endl;
		data_roi++;
	}*/

	vector<float> result;
	for (int k = 0; k < res_size/3; ++k) {
		float u = *data_guided;
		++data_guided;
		float v = *data_guided;
		++data_guided;
		float d = *data_guided;
		++data_guided;
		// transform
		if (is_crop) {
			u = (u+1) / 2 * cv_img.cols;
			v = (v+1) / 2 * cv_img.rows;
			//std::cout << u << " " << v << " " << d << std::endl;
		}
		else {
			float c1 = center[0];
			float c2 = center[1];
			float c3 = center[2];
			// std::cout << c1 << " " << c2 << " " << c3 << std::endl;
			u = u * cube_length[0] * fabs(fx) / c3 + c1;
			v = v * cube_length[1] * fabs(fy) / c3 + c2;
			d = d * cube_length[2] + c3;
			//std::cout << u << " " << v << " " << d << std::endl;
		}
		result.push_back(u);
		result.push_back(v);
		result.push_back(d);
	}

	return result;
}

vector<float> HandPoseEstimator::predict(const cv::Mat& cv_img, cv::Mat& crop) {
	// get outputs and save
	const boost::shared_ptr<Blob<float> > blob =
		test_net->blob_by_name(output_blob);
	int batch_size = blob->num();
	int res_size = blob->count() / batch_size;
	// pre processing
	vector<float> center;
	//cv::imshow("cv_img", cv_img);

	get_center(cv_img, center, lower, upper);
	crop = crop_image(cv_img, center, cube_length, fx, fy, height, width);
	//cv::imshow("crop", crop);

	//feed data
	boost::shared_ptr<Blob<float> > blob_data = test_net->blob_by_name("data");
	Mat2Blob(crop, blob_data);
	// forward
	test_net->Forward();
	const float* data = blob->cpu_data();

	vector<float> result;
	float c1 = center[0];
	float c2 = center[1];
	float c3 = center[2];
	std::cout << c1 << " " << c2 << " " << c3 << std::endl;

	for (int k = 0; k < res_size / 3; ++k) {
		float u = *data;
		++data;
		float v = *data;
		++data;
		float d = *data;
		++data;
		// transform
		u = u * cube_length[0] * fabs(fx) / c3 + c1;
		v = v * cube_length[1] * fabs(fy) / c3 + c2;
		d = d * cube_length[2] + c3;
		//std::cout << u << " " << v << " " << d << std::endl;

		result.push_back(u);
		result.push_back(v);
		result.push_back(d);
	}

	return result;
}


void HandPoseEstimator::Mat2Blob(const cv::Mat &mat,
	boost::shared_ptr<Blob<float> > blob) {
	assert(!mat.empty());
	float* blob_data = blob->mutable_cpu_data();

	int height = mat.rows;
	int width = mat.cols;
	int channel = 1;
	int top_index = 0;

	for (int c = 0; c < channel; ++c)
	{
		// cout << mat[c] << endl;
		for (int h = 0; h < height; ++h)
		{
			const float* ptr = mat.ptr<float>(h);
			for (int w = 0; w < width; ++w)
			{
				// int top_index = (c * height + h) * width + w;
				blob_data[top_index++] = ptr[w];
			}
		}
	}
}

void HandPoseEstimator::Pose2Blob(const float* guided_pose, int res_size, boost::shared_ptr<Blob<float> > blob) {
	float* blob_data = blob->mutable_cpu_data();

	int top_index = 0;

	for (int h = 0; h < res_size; ++h)
	{
		blob_data[top_index++] = guided_pose[h];
	}
}

void HandPoseEstimator::get_center(const cv::Mat& cv_img, vector<float>& center, int lower, int upper) {
	// TODO(guohengkai): remove the hard threshold if necessary 0 ~ 880
	center = vector<float>(3, 0);
	int count = 0;
	int min_val = INT_MAX;
	int max_val = INT_MIN;
	for (int r = 0; r < cv_img.rows; ++r) {
		const float* ptr = cv_img.ptr<float>(r);
		for (int c = 0; c < cv_img.cols; ++c) {
			//std::cout << ptr[c] << std::endl;
			if (ptr[c] <= upper && ptr[c] >= lower) {
				center[0] += c;
				center[1] += r;
				center[2] += ptr[c];
				++count;
			}
			/*
			if (int(ptr[c]) > 0)
			min_val = std::min(int(ptr[c]), min_val);
			max_val = std::max(int(ptr[c]), max_val);
			*/
		}
	}
	if (count) {
		for (int i = 0; i < 3; ++i) {
			center[i] /= count;
		}
		//std::cout << center[0] << " " << center[1] << " " << center[2] << std::endl;
	}
	else
	{
		center.clear();
		// LOG(INFO) << "max: " << max_val << ", min: " << min_val;
	}
}

cv::Mat HandPoseEstimator::crop_image(const cv::Mat& cv_img,
	const vector<float>& center, const vector<int>& cube_length,
	float fx, float fy, int height, int width) {
	float xstart = center[0] - cube_length[0] / center[2] * fabs(fx);
	float xend = center[0] + cube_length[0] / center[2] * fabs(fx);
	float ystart = center[1] - cube_length[1] / center[2] * fabs(fy);
	float yend = center[1] + cube_length[1] / center[2] * fabs(fy);
	float xscale = 2.0 / (xend - xstart);
	float yscale = 2.0 / (yend - ystart);
	//std::cout << "crop:" << xstart << " " << xend << " " << ystart << " " << yend << std::endl;
	//std::cout << "cube:" << cube_length[0] << " " << cube_length[1] << std::endl;

	vector<cv::Point2f> src, dst;
	src.push_back(cv::Point2f(xstart, ystart));
	dst.push_back(cv::Point2f(0, 0));
	src.push_back(cv::Point2f(xstart, yend));
	dst.push_back(cv::Point2f(0, height - 1));
	src.push_back(cv::Point2f(xend, ystart));
	dst.push_back(cv::Point2f(width - 1, 0));
	cv::Mat trans = cv::getAffineTransform(src, dst);
	cv::Mat res_img;
	cv::warpAffine(cv_img, res_img, trans, cv::Size(width, height),
		cv::INTER_LINEAR, cv::BORDER_CONSTANT, center[2] + cube_length[2]);
	res_img -= center[2];
	res_img = cv::max(res_img, -cube_length[2]);
	res_img = cv::min(res_img, cube_length[2]);
	res_img /= cube_length[2];
	return res_img;
}

void HandPoseEstimator::get_skeleton_setting(string dataset, vector<int>& joint_id_start, vector<int>& joint_id_end) {
	if (dataset.compare("icvl") == 0) {
		joint_id_start = { 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14 };
		joint_id_end = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}
	else if (dataset.compare("hands17") == 0) {
		joint_id_start = { 0, 0, 0, 0, 0, 1, 6, 7, 2, 9, 10, 3, 12, 13, 4, 15, 16, 5, 18, 19 };
		joint_id_end = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
	}
	else if (dataset.compare("nyu") == 0) {
		joint_id_start = { 0,  0,  0,  3,  4,  0,  6,  0,  8,  0, 10,  0, 12 };
		joint_id_end = { 1,  2,  5,  4,  5,  7,  7,  9,  9, 11, 11, 13, 13 };
	}
	else if (dataset.compare("msra") == 0) {
		joint_id_start = {0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11,  0, 13, 14, 15, 0, 17, 18, 19};
		joint_id_end = {1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
	}
}

void HandPoseEstimator::get_sketch_color(string dataset, vector<vector<int>>& color_bone) {
    if (dataset.compare("icvl") == 0) {
        color_bone = {RED, RED, RED, GREEN, GREEN, GREEN,
                BLUE, BLUE, BLUE, YELLOW, YELLOW, YELLOW,
                PURPLE, PURPLE, PURPLE};
    }
    else if (dataset.compare("nyu") == 0) {
        color_bone = {CYAN, CYAN, RED, RED, RED, GREEN, GREEN,
                BLUE, BLUE, YELLOW, YELLOW, PURPLE, PURPLE};
    }
    else if (dataset.compare("msra") == 0) {
        color_bone = {RED, RED, RED, RED, GREEN, GREEN, GREEN, GREEN,
                BLUE, BLUE, BLUE, BLUE, YELLOW, YELLOW, YELLOW, YELLOW,
                PURPLE, PURPLE, PURPLE, PURPLE};
    }
    else if (dataset.compare("hands17") == 0) {
        color_bone = {GREEN, BLUE, YELLOW, PURPLE, RED, 
              GREEN, GREEN, GREEN,
              BLUE, BLUE, BLUE,
              YELLOW, YELLOW, YELLOW,
              PURPLE, PURPLE, PURPLE,
              RED, RED, RED};
    }
}
        
void HandPoseEstimator::get_joint_color(string dataset, vector<vector<int>>& color_joint) {
    if (dataset.compare("icvl") == 0) {
        color_joint = {CYAN, RED, RED, RED, GREEN, GREEN, GREEN,
                BLUE, BLUE, BLUE, YELLOW, YELLOW, YELLOW,
                PURPLE, PURPLE, PURPLE};
    }
    else if (dataset.compare("nyu") == 0) {
        color_joint = {CYAN, CYAN, CYAN, RED, RED, RED, GREEN, GREEN,
                BLUE, BLUE, YELLOW, YELLOW, PURPLE, PURPLE};
    }
    else if (dataset.compare("msra") == 0) {
        color_joint = {CYAN, RED, RED, RED, RED, GREEN, GREEN, GREEN, GREEN,
                BLUE, BLUE, BLUE, BLUE, YELLOW, YELLOW, YELLOW, YELLOW,
                PURPLE, PURPLE, PURPLE, PURPLE};
    }
    else if (dataset.compare("hands17") == 0) {
        color_joint = {CYAN, GREEN, BLUE, YELLOW, PURPLE, RED,
                    GREEN, GREEN, GREEN,
                    BLUE, BLUE, BLUE,
                    YELLOW, YELLOW, YELLOW,
                    PURPLE, PURPLE, PURPLE,
                    RED, RED, RED};
    }
}

