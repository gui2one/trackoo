#pragma once

#define DLIB_USE_CUDA

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <opencv2/opencv.hpp>


#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>





#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <dlib/cuda/gpu_data.h>
#include <dlib/cuda/cuda_dlib.h>
#include <dlib/cuda/cuda_utils.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "ofMain.h"
#include "ofxCv.h"


struct TransformVectors {
	ofVec3f translates;
	ofVec3f rotates;
};

class gui2oneFaceDetectorInstance : public cv::Rect {

public:
	TransformVectors vectors;

};

class MyFollower : public ofxCv::RectFollower, gui2oneFaceDetectorInstance
{
	public:
		// code here
};

class Gui2oneFaceDetector
{
public:
	Gui2oneFaceDetector(int w = 1920, int h = 1080);
	
	~Gui2oneFaceDetector();

	void setProcessSize(int w, int h);
	void initCvDnnNet(std::string  proto, std::string caffe_model);
	void initDlibShapePredictor(std::string landmarks_model);

	void initEsitmateTransforms();
	std::vector<dlib::rectangle> detectFaces(cv::Mat& frame);

	std::vector<dlib::full_object_detection> detectLandmarks(std::vector<dlib::rectangle>& _rectangles, cv::Mat _frame);
	dlib::full_object_detection              detectLandmarks(dlib::rectangle& _rectangle, cv::Mat _frame);

	std::vector<ofPolyline> getPolylines(dlib::full_object_detection & detection);

	ofMesh getMesh(dlib::full_object_detection & detection);

	void cvRenderFacesLandmarks(cv::Mat _frame, std::vector<dlib::full_object_detection>& shapes);
	
	
	std::vector<TransformVectors> estimateTransforms(const std::vector<dlib::full_object_detection>& detections, std::vector<dlib::rectangle> rectangles, cv::Mat& _frame,float desired_aov, bool draw_infos = true);
	TransformVectors              estimateTransforms(const dlib::full_object_detection& detection, dlib::rectangle rectangle, cv::Mat& _frame,float desired_aov, bool draw_infos = true);

	int proc_width, proc_height;

	dlib::shape_predictor pose_model;

	
	std::vector<cv::Vec3d> object_points2;
	std::vector<cv::Vec2d> image_points2;

	
	cv::dnn::experimental_dnn_34_v11::Net m_dnn_net;
	

	
	

		
};

