#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>



#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "ofMain.h"

struct TransformVectors {
	ofVec3f translates;
	ofVec3f rotates;
};
class Gui2oneFaceDetector
{
public:
	Gui2oneFaceDetector(int w = 640, int h = 480);
	
	~Gui2oneFaceDetector();

	void setProcessSize(int w, int h);
	void initCvDnnNet(std::string  proto, std::string caffe_model);
	void initDlibShapePredictor(std::string landmarks_model);

	void initEsitmateTransforms();
	std::vector<dlib::rectangle> detectFaces(cv::Mat& frame);
	std::vector<dlib::full_object_detection> detectLandmarks(std::vector<dlib::rectangle>& _rectangles, cv::Mat _frame);


	void cvRenderFacesLandmarks(cv::Mat _frame, std::vector<dlib::full_object_detection>& shapes);
	
	std::vector<TransformVectors> estimateTransforms(const std::vector<dlib::full_object_detection>& detections, std::vector<dlib::rectangle> rectangles, cv::Mat& _frame,float desired_aov, bool draw_infos = true);

	int proc_width, proc_height;

	dlib::shape_predictor pose_model;

	cv::Mat object_points; 
	cv::Mat image_points; 
	
	std::vector<cv::Vec3d> object_points2;
	std::vector<cv::Vec2d> image_points2;

	
	cv::dnn::experimental_dnn_34_v11::Net m_dnn_net;

		
};

