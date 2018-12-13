// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

/////////////////////////////////////////////////////
// librealsense tutorial #1 - Accessing depth data //
/////////////////////////////////////////////////////

// First include the librealsense C++ header file
#include <librealsense2/rs.hpp>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "hand_pose_estimator.h"
#include <ctime>

#define RUN_BASELINE 0
#define RUN_POSE_REN 1


cv::Mat show_depth_joints(cv::Mat dst, cv::Mat crop, vector<float> result, HandPoseEstimator hpe, 
	string dataset, bool is_crop=false) {
	// convert depth image
	cv::Mat show(dst.clone());
	show.setTo(10000, show == 0);
	cv::threshold(show, show, 1000, 1000, cv::THRESH_TRUNC);
	double minVal, maxVal;
	cv::minMaxLoc(show, &minVal, &maxVal);
	show.convertTo(show, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
	
	cv::Rect roi;
	if (!is_crop) {
		// put crop image on the left corner
		crop = (crop + 1) * 255 / 2;
		if (dataset.compare("hands17") == 0)
			roi = cv::Rect(show.cols - 96, 0, 96, 96);
		else
			roi = cv::Rect(0, 0, 96, 96);
		crop.copyTo(show(roi));
	}
	int joint_size = 8;
	int bone_size = 3;
	if (is_crop) {
		joint_size = 1;
		bone_size = 1;
	}
	// draw joints
    vector<vector<int>> color_joint;
    hpe.get_joint_color(dataset, color_joint);
	cv::cvtColor(show, show, CV_GRAY2BGR);
	for (int i = 0; i < result.size() / 3; ++i) {
		//cv::circle(show, cv::Point2f(result[i * 3], result[i * 3 + 1]), joint_size,
		//	cv::Scalar(0, 0, 255), -1);
        vector<int> c = color_joint[i];
		cv::circle(show, cv::Point2f(result[i * 3], result[i * 3 + 1]), joint_size,
			cv::Scalar(c[0], c[1], c[2]), -1);
	}
	// draw bones
	vector<int> joint_id_start;
	vector<int> joint_id_end;
	hpe.get_skeleton_setting(dataset, joint_id_start, joint_id_end);
    vector<vector<int>> color_bone;
    hpe.get_sketch_color(dataset, color_bone);
	// draw bone
	for (int i = 0; i < joint_id_start.size(); i++) {
		int k1 = joint_id_start[i];
		int k2 = joint_id_end[i];
        vector<int> c = color_bone[i];
		//cv::line(show, cv::Point2f(result[k1 * 3], result[k1 * 3 + 1]),
		//	cv::Point2f(result[k2 * 3], result[k2 * 3 + 1]),
		//	cv::Scalar(0, 255, 0), bone_size);
		cv::line(show, cv::Point2f(result[k1 * 3], result[k1 * 3 + 1]),
			cv::Point2f(result[k2 * 3], result[k2 * 3 + 1]),
			cv::Scalar(c[0], c[1], c[2]), bone_size);
	}
	if (!is_crop) {
		// draw crop rect
		cv::rectangle(show, roi, cv::Scalar(255, 0, 0), 2);
	}
	// flip or not
	cv::Mat showf(show.clone());
	if (dataset.compare("hands17") == 0)
		cv::flip(show, showf, 1);
	return showf;
}

float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

int main(int argc, char** argv) try
{
	// caffe model
	string dataset = "hands17";
	bool is_save_result = false;
	string save_dir = "results";
	if (argc >= 2)
	    dataset = string(argv[1]);
    if (argc >= 3) {
        save_dir = string(argv[2]);
        is_save_result = true;
    }
	HandPoseEstimator hpe(dataset);

    // Create a context object. This object owns the handles to all connected realsense devices.
    //rs::context ctx;
    //printf("There are %d connected RealSense devices.\n", ctx.get_device_count());
    //if (ctx.get_device_count() == 0) return EXIT_FAILURE;

    // This tutorial will access only a single device, but it is trivial to extend to multiple devices
    //rs::device * dev = ctx.get_device(0);
    //printf("\nUsing device 0, an %s\n", dev->get_name());
    //printf("    Serial number: %s\n", dev->get_serial());
    //printf("    Firmware version: %s\n", dev->get_firmware_version());

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    rs2::pipeline_profile profile = pipe.start();

    const float depth_scale = get_depth_scale(profile.get_device());

    int cnt = 0;
    while (true)
    {
        // This call waits until a new coherent set of frames is available on a device
        // Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        //pipe.wait_for_frames();

        // get depth
        rs2::frameset data = pipe.wait_for_frames();
        cv::Mat depth(cv::Size(640, 480), CV_16UC1, (void*)data.get_depth_frame().get_data(), cv::Mat::AUTO_STEP);
        depth.convertTo(depth, CV_32FC1);
        depth = depth * depth_scale * 1000.0f;
        depth.setTo(10000, depth == 0);

        //std::cout << dev->get_depth_scale() << std::endl;
        cv::Mat dst = depth;
        cv::Mat crop;
        if (dataset.compare("icvl") == 0)
            cv::flip(depth, dst, 1);
        // cv::imshow("dst", dst);

        if (RUN_BASELINE) {
            vector<float> result = hpe.predict(dst, crop);
            // display
            cv::Mat show = show_depth_joints(dst, crop.clone(), result, hpe, dataset);
            cv::imshow("result", show);
        }
        if (RUN_POSE_REN) {
		    std::clock_t start;
		    double duration;
		    start = std::clock();
            vector<float> result_guided = hpe.predict_guided(dst, crop);
		    duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		    std::cout << "overall time: " << duration << '\n';
            cv::Mat show_guided = show_depth_joints(dst, crop.clone(), result_guided, hpe, dataset);
            cv::imshow("result_guided", show_guided);
            if (is_save_result) {
                string filename = save_dir + "/result_guided_";
                filename += std::to_string(cnt) + ".png";
                std::cout << "saving results to " << filename << std::endl;
                cv::imwrite(filename, show_guided);
            }
        }

        char kb = cv::waitKey(1);
        if(kb == 'q')
            break;

        cnt++;
    }

	return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
	// Method calls against librealsense objects may throw exceptions of type rs::error
	printf("rs::error was thrown when calling %s(%s):\n", e.get_failed_function().c_str(), e.get_failed_args().c_str());
	printf("    %s\n", e.what());
	return EXIT_FAILURE;
}


