#include "gui2one_face_detector.h"



Gui2oneFaceDetector::Gui2oneFaceDetector()
{
	initCvDnnNet("./data/deploy.prototxt.txt", "./data/res10_300x300_ssd_iter_140000.caffemodel");
	initDlibShapePredictor("./data/shape_predictor_68_face_landmarks.dat");
	initEsitmateTransforms();
}


Gui2oneFaceDetector::~Gui2oneFaceDetector()
{
}

void Gui2oneFaceDetector::initCvDnnNet(std::string proto, std::string caffe_model)
{
	// "./data/deploy.prototxt.txt", "./data/res10_300x300_ssd_iter_140000.caffemodel"
	m_dnn_net = cv::dnn::readNetFromCaffe(proto, caffe_model);
}

void Gui2oneFaceDetector::initDlibShapePredictor(std::string landmarks_model)
{
	// Load pose estimation model.

	dlib::deserialize(landmarks_model) >> pose_model;
}


void Gui2oneFaceDetector::initEsitmateTransforms()
{
	object_points = cv::Mat(68, 3, CV_32F);
	image_points = cv::Mat(68, 2, CV_32F);

	// set rest landmarks 2D positions
	// 12 points
	// 0 4 12 16 --> jaw
	// 17 21     --> right eyebrow
	// 22 26     --> right eyebrow
	// 27 30     --> nose line
	// 48 54     --> mouth outside corners
	
		//// houdini generated
	object_points2.push_back(cv::Vec3d(0.061, -0.536, 0.265));
	object_points2.push_back(cv::Vec3d(0.168, -0.340, 0.725));
	object_points2.push_back(cv::Vec3d(0.954, -0.340, 0.725));
	object_points2.push_back(cv::Vec3d(1.061, -0.536, 0.265));
	object_points2.push_back(cv::Vec3d(0.193, -0.259, 0.099));
	object_points2.push_back(cv::Vec3d(0.457, -0.104, 0.071));
	object_points2.push_back(cv::Vec3d(0.665, -0.104, 0.071));
	object_points2.push_back(cv::Vec3d(0.929, -0.259, 0.099));
	object_points2.push_back(cv::Vec3d(0.561, -0.098, 0.186));
	object_points2.push_back(cv::Vec3d(0.562, -0.006, 0.384));
	object_points2.push_back(cv::Vec3d(0.389, -0.137, 0.650));
	object_points2.push_back(cv::Vec3d(0.741, -0.149, 0.662));
	
}

std::vector<dlib::rectangle> Gui2oneFaceDetector::detectFaces(cv::Mat& frame)
{


	//printf("frame width : %d\n", frame.size().width);
			
	
	cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(640, 360), 100, false, false);
	m_dnn_net.setInput(inputBlob, "data");
	cv::Mat detection = m_dnn_net.forward("detection_out");
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());



	std::vector<dlib::rectangle> rectangles;
	rectangles.clear();
	for (int i = 0; i < detectionMat.rows; i++)
	{

		float confidence = detectionMat.at<float>(i, 2);
		float confidenceThreshold = 0.5;
		int frameWidth = frame.cols;
		int frameHeight = frame.rows;
		if (confidence > confidenceThreshold)
		{
			//printf("Confidence : %.3f\n", confidence);
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);


			//cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
			dlib::point p1 = dlib::point(x1, y1);
			dlib::point p2 = dlib::point(x2, y2);
			dlib::rectangle rect = dlib::rectangle(p1, p2);


			rectangles.push_back(rect);
			//frame = frame(rect);
			//cv::resizeWindow("edges", rect.size());
		}
	}

	return rectangles;
}

std::vector<dlib::full_object_detection> Gui2oneFaceDetector::detectLandmarks
(
	std::vector<dlib::rectangle>& _rectangles,
	cv::Mat _frame
)
{
	// convert cv::Mat to dlib format
	dlib::cv_image<dlib::bgr_pixel> cimg(_frame);

	std::vector<dlib::full_object_detection> shapes;
	for (unsigned long i = 0; i < _rectangles.size(); ++i)
	{
		//printf("Face %d\n", i);
		shapes.push_back(pose_model(cimg, _rectangles[i]));
	}
	return shapes;
}

void Gui2oneFaceDetector::cvRenderFacesLandmarks(cv::Mat _frame , std::vector<dlib::full_object_detection>& shapes)
{
	

	for (size_t i = 0; i < shapes.size(); i++)
	{
		dlib::full_object_detection& shape = shapes[i];
		cv::drawMarker(_frame, cv::Point(shape.part(0).x(), shape.part(0).y()), cv::Scalar(20, 255, 255, 255));
		cv::drawMarker(_frame, cv::Point(shape.part(16).x(), shape.part(16).y()), cv::Scalar(20, 255, 255, 255));
		cv::drawMarker(_frame, cv::Point(shape.part(36).x(), shape.part(36).y()), cv::Scalar(20, 255, 255, 255));
		cv::drawMarker(_frame, cv::Point(shape.part(45).x(), shape.part(45).y()), cv::Scalar(20, 255, 255, 255));
	}

}

std::vector<TransformVectors> Gui2oneFaceDetector::estimateTransforms(const std::vector<dlib::full_object_detection>& dets, std::vector<dlib::rectangle> rectangles, cv::Mat& _frame, float desired_aov, bool draw_infos)
{
	
	//object_points2.clear();
	std::vector<ofMatrix4x4> matrices;
	std::vector<TransformVectors> tr_vectors;

	for (int det_id = 0; det_id < dets.size(); det_id++) {

		const dlib::full_object_detection& d0 = dets[det_id];
		image_points2.clear();

		for (unsigned long i = 0; i < d0.num_parts(); i++) {

			const dlib::point& pt = d0.part(i);

			object_points.at<float>((int)i, 0) = (float)pt.x();
			object_points.at<float>((int)i, 1) = (float)pt.y();
			object_points.at<float>((int)i, 2) = (float)pt.z();

		}

		image_points2.push_back(cv::Vec2d(d0.part( 0).x() , d0.part( 0).y()));
		image_points2.push_back(cv::Vec2d(d0.part (4).x() , d0.part( 4).y()));
		image_points2.push_back(cv::Vec2d(d0.part(12).x() , d0.part(12).y()));
		image_points2.push_back(cv::Vec2d(d0.part(16).x() , d0.part(16).y()));
		image_points2.push_back(cv::Vec2d(d0.part(17).x() , d0.part(17).y()));
		image_points2.push_back(cv::Vec2d(d0.part(21).x() , d0.part(21).y()));
		image_points2.push_back(cv::Vec2d(d0.part(22).x() , d0.part(22).y()));
		image_points2.push_back(cv::Vec2d(d0.part(26).x() , d0.part(26).y()));
		image_points2.push_back(cv::Vec2d(d0.part(27).x() , d0.part(27).y()));
		image_points2.push_back(cv::Vec2d(d0.part(30).x() , d0.part(30).y()));
		image_points2.push_back(cv::Vec2d(d0.part(48).x() , d0.part(48).y()));
		image_points2.push_back(cv::Vec2d(d0.part(54).x() , d0.part(54).y()));

		// make camera matrix
		cv::Mat cam_mat = cv::Mat::eye(3, 3, CV_64F);
		cam_mat.at<double>(0, 0) = 640.0;
		cam_mat.at<double>(1, 0) = 0.0;
		cam_mat.at<double>(2, 0) = 640.0/2.0;

		cam_mat.at<double>(0, 1) = 0.0;
		cam_mat.at<double>(1, 1) = 360.0;
		cam_mat.at<double>(2, 1) = 360.0/2.0;

		cam_mat.at<double>(0, 2) = 0.0;
		cam_mat.at<double>(1, 2) = 0.0;
		cam_mat.at<double>(2, 2) = 1.0;


		float aov = desired_aov;
		float focalLength = 640.0 * ofDegToRad(aov);
		float opticalCenterX = 640.0 / 2.0;
		float opticalCenterY = 360.0 / 2.0;

		cv::Mat1d projectionMat = cv::Mat::zeros(3, 3, CV_32F);
		
		projectionMat(0, 0) = focalLength;
		projectionMat(1, 1) = focalLength;
		projectionMat(0, 2) = opticalCenterX;
		projectionMat(1, 2) = opticalCenterY;
		projectionMat(2, 2) = 1.0;
		//cv::Mat dist_coeffs(0, 0, CV_64F); // = { 0.0 }; // can leave empty if no camera distortion ...otherwise do a calibration ?

		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); // = { 0.0, 0.0, 0.0 };
		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);


		cv::solvePnP(object_points2, image_points2, projectionMat, cv::noArray(), rvec, tvec, false , cv::SOLVEPNP_EPNP);
		//cv::solvePnP(object_points2, image_points2, projectionMat, cv::noArray(), rvec, tvec, false , cv::SOLVEPNP_UPNP);

		// Black magic: The x axis in the rotation vector needs to get flipped.
		double * r = rvec.ptr<double>(0);
		r[0] *= -1;
		r[1] *= -1;

		TransformVectors result;
		result.translates = ofVec3f(tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2));
		result.rotates = ofVec3f(rvec.at<double>(0, 0), rvec.at<double>(0, 1), rvec.at<double>(0, 2));
		
		tr_vectors.push_back(result);

		if (draw_infos)
		{
			// display Data on Image
			float font_size = 0.5;
			int font_thickness = 1;
			cv::Scalar font_color(255, 255, 255);


			int font_face = CV_FONT_HERSHEY_SIMPLEX;

			char buff[255];
			sprintf(buff, "Face %d", det_id);
			cv::Size text_size = cv::getTextSize(buff, font_face, font_size, font_thickness, 0);

			cv::Point text_pos = cv::Point(rectangles[det_id].tl_corner().x(), rectangles[det_id].tl_corner().y());
			cv::Point text_end = cv::Point(text_pos.x + text_size.width, (text_pos.y - text_size.height));
			cv::rectangle(_frame, text_pos, text_end, cv::Scalar(0, 0, 0, 0), CV_FILLED);
			cv::putText(_frame, buff, text_pos, CV_FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness);

			char buff_rot[255];
			sprintf(buff_rot, "rotation: %.3f %.3f %.3f", rvec.at<double>(0, 0), rvec.at<double>(0, 1), rvec.at<double>(0, 2));
			cv::Size text_size_rot = cv::getTextSize(buff_rot, font_face, font_size, font_thickness, 0);

			cv::Point text_pos_rot = cv::Point(rectangles[det_id].tl_corner().x(), rectangles[det_id].tl_corner().y() + 20);
			cv::Point text_end_rot = cv::Point(text_pos_rot.x + text_size_rot.width, text_pos_rot.y - text_size_rot.height);
			cv::rectangle(_frame, text_pos_rot, text_end_rot, cv::Scalar(0, 0, 0, 0), CV_FILLED);
			cv::putText(_frame, buff_rot, text_pos_rot, CV_FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness);

			char buff_trans[255];
			sprintf(buff_trans, "translation: %.3f %.3f %.3f", tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2));
			cv::Size text_size_trans = cv::getTextSize(buff_trans, font_face, font_size, font_thickness, 0);

			cv::Point text_pos_trans = cv::Point(rectangles[det_id].tl_corner().x(), rectangles[det_id].tl_corner().y() + 50);
			cv::Point text_end_trans = cv::Point(text_pos_trans.x + text_size_trans.width, text_pos_trans.y - text_size_trans.height);
			cv::rectangle(_frame, text_pos_trans, text_end_trans, cv::Scalar(0, 0, 0, 0), CV_FILLED);
			cv::putText(_frame, buff_trans, text_pos_trans, CV_FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness);


		}

		// create transform Matrix
		//ofMatrix4x4 of_matrix = ofMatrix4x4();

		//of_matrix.rotate(rvec.at<double>(0, 0), 1.0, 0.0, 0.0);
		//of_matrix.rotate(rvec.at<double>(0, 1), 0.0, 1.0, 0.0);
		//of_matrix.rotate(rvec.at<double>(0, 2), 0.0, 0.0, 1.0);
		//of_matrix.translate(tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2));

		//

		//

		//matrices.push_back(of_matrix);
	}

	return tr_vectors;
}
