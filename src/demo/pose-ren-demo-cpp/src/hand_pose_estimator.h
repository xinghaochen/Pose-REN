#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

//#include <gflags/gflags.h>
//#include <glog/logging.h>
#include <cstdio>
#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/opencv.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
using std::vector;

#define SQR(x) ((x)*(x))

class HandPoseEstimator {
public:
	HandPoseEstimator(string dataset = "hands17");
	~HandPoseEstimator();

	int gpu;		//Optional; run in GPU mode on given device ID. -1 for CPU.
	string model;	// "The model definition protocol buffer text file.");
	string weights;	// "The weights to initialize model.");
	string model_guided;	// "The model definition protocol buffer text file.");
	string weights_guided;	// "The weights to initialize model.");
	string phase;	// "Network phase (train or test).");
	string output_blob_init;	// "Blob name for output.")
	string output_blob;	// "Blob name for output.")

	double fx;
	double fy;
	double ux;
	double uy;
	int lower;
	int upper;
	vector<int> cube_length;
	int height, width;

	boost::shared_ptr<Net<float> > test_net;
	boost::shared_ptr<Net<float> > test_net_guided;

	void get_skeleton_setting(string dataset, vector<int>& joint_id_start, vector<int>& joint_id_end);
    void get_sketch_color(string dataset, vector<vector<int>>& color_bone);
    void get_joint_color(string dataset, vector<vector<int>>& color_joint);

	int init_model();
	vector<float> predict(const cv::Mat& cv_img, cv::Mat& crop);
	vector<float> predict_guided(const cv::Mat& cv_img, cv::Mat& crop, bool is_crop = false);

	void Mat2Blob(const cv::Mat &mat,
		boost::shared_ptr<Blob<float> > blob);

	void Pose2Blob(const float* guided_pose, int res_size, boost::shared_ptr<Blob<float> > blob);

	void get_center(const cv::Mat& cv_img, vector<float>& center, int lower, int upper);

	cv::Mat crop_image(const cv::Mat& cv_img,
		const vector<float>& center, const vector<int>& cube_length,
		float fx, float fy, int height, int width);
private:
    vector<int> RED = {0, 0, 255};
    vector<int> GREEN = {75, 255, 66};
    vector<int> BLUE = {255, 64, 0};
    vector<int> YELLOW = {17, 240, 244};
    vector<int> PURPLE = {255, 255, 0};
    vector<int> CYAN = {255, 0, 255};
};
