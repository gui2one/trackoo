#include "gui2one_face_detector.h"



Gui2oneFaceDetector::Gui2oneFaceDetector(int w, int h)
{

	proc_width = w;
	proc_height = h;

	initCvDnnNet("./data/deploy.prototxt.txt", "./data/res10_300x300_ssd_iter_140000.caffemodel");
	initDlibShapePredictor("./data/shape_predictor_68_face_landmarks.dat");
	initEsitmateTransforms();
}

void Gui2oneFaceDetector::setProcessSize(int w, int h) 
{
	proc_width = w;
	proc_height = h;
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

	// set rest landmarks 2D positions
	// 12 points
	// 0 4 12 16 --> jaw
	// 17 21     --> right eyebrow
	// 22 26     --> right eyebrow
	// 27 30     --> nose line
	// 48 54     --> mouth outside corners
	
		//// houdini generated
	object_points2.push_back(cv::Vec3d(0.012, -0.536, 0.265)); //point id 0
	object_points2.push_back(cv::Vec3d(0.116, -0.340, 0.725)); //point id 4
	object_points2.push_back(cv::Vec3d(0.882, -0.340, 0.725)); //point id 12
	object_points2.push_back(cv::Vec3d(0.986, -0.536, 0.265)); //point id 16
	object_points2.push_back(cv::Vec3d(0.141, -0.259, 0.099)); //point id 17
	object_points2.push_back(cv::Vec3d(0.398, -0.104, 0.071)); //point id 21
	object_points2.push_back(cv::Vec3d(0.601, -0.104, 0.071)); //point id 22
	object_points2.push_back(cv::Vec3d(0.858, -0.259, 0.099)); //point id 26
	object_points2.push_back(cv::Vec3d(0.499, -0.098, 0.186)); //point id 27
	object_points2.push_back(cv::Vec3d(0.500, -0.007, 0.384)); //point id 30
	object_points2.push_back(cv::Vec3d(0.188, -0.258, 0.209)); //point id 36
	object_points2.push_back(cv::Vec3d(0.386, -0.210, 0.216)); //point id 39
	object_points2.push_back(cv::Vec3d(0.612, -0.210, 0.216)); //point id 42
	object_points2.push_back(cv::Vec3d(0.810, -0.258, 0.209)); //point id 45
	object_points2.push_back(cv::Vec3d(0.332, -0.138, 0.650)); //point id 48
	object_points2.push_back(cv::Vec3d(0.675, -0.149, 0.662)); //point id 54


	
}

std::vector<dlib::rectangle> Gui2oneFaceDetector::detectFaces(cv::Mat& frame)
{

			
	
	
	cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(proc_width, proc_height), 300, false, false);
	m_dnn_net.setInput(inputBlob, "data");
	cv::Mat detection = m_dnn_net.forward("detection_out");
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());



	std::vector<dlib::rectangle> rectangles;
	rectangles.clear();
	for (int i = 0; i < detectionMat.rows; i++)
	{

		float confidence = detectionMat.at<float>(i, 2);
		float confidenceThreshold = 0.9;
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

dlib::full_object_detection Gui2oneFaceDetector::detectLandmarks(dlib::rectangle & _rectangle, cv::Mat _frame)
{
	// convert cv::Mat to dlib format
	dlib::cv_image<dlib::bgr_pixel> cimg(_frame);


	dlib::full_object_detection shape;

	shape = pose_model(cimg, _rectangle);

	return shape;
	
}

std::vector<ofPolyline> Gui2oneFaceDetector::getPolylines(dlib::full_object_detection& detection) {

	


	std::vector<ofPolyline> polylines;


	
	
	ofPolyline jaw_line;

	//// jaw line
	for (size_t i = 0; i <= 16; i++)
	{
		jaw_line.addVertex(detection.part(i).x(), detection.part(i).y(), 0.0f);
	}
	polylines.push_back(jaw_line);
	//// right_eye_brow
	ofPolyline right_eye_brow;
	for (size_t i = 17; i <= 21; i++)
	{
		right_eye_brow.addVertex(detection.part(i).x(), detection.part(i).y(), 0.0f);
	}
	polylines.push_back(right_eye_brow);

	//// left_eye_brow
	ofPolyline left_eye_brow;
	for (size_t i = 22; i <= 26; i++)
	{
		left_eye_brow.addVertex(detection.part(i).x(), detection.part(i).y(), 0.0f);
	}
	polylines.push_back(left_eye_brow);

	//// nose_line
	ofPolyline nose_line;
	for (size_t i = 27; i <= 30; i++)
	{
		nose_line.addVertex(detection.part(i).x(), detection.part(i).y(), 0.0f);
	}
	polylines.push_back(nose_line);

	//// nose_line_2
	ofPolyline nose_line_2;
	for (size_t i = 31; i <= 35; i++)
	{
		nose_line_2.addVertex(detection.part(i).x(), detection.part(i).y(), 0.0f);
	}
	polylines.push_back(nose_line_2);

	//// right_eye
	ofPolyline right_eye;
	for (size_t i = 36; i <= 41; i++)
	{
		right_eye.addVertex(detection.part(i).x(), detection.part(i).y(), 0.0f);
	}
	polylines.push_back(right_eye);

	//// left_eye
	ofPolyline left_eye;
	for (size_t i = 42; i <= 47; i++)
	{
		left_eye.addVertex(detection.part(i).x(), detection.part(i).y(), 0.0f);
	}
	polylines.push_back(left_eye);
	

	return polylines;
}

ofMesh Gui2oneFaceDetector::getMesh(dlib::full_object_detection& detection) {

	ofMesh mesh;
	
	const dlib::full_object_detection d = detection;
	std::vector<glm::vec3> verts;
	//verts.reserve(d.num_parts());
	for (size_t part_id = 0; part_id < 60; part_id++) {

		verts.push_back(glm::vec3(d.part(part_id).x(), d.part(part_id).y(), 0.0));


	}
	


	// adding forhead vertices
	glm::vec3 pos1, pos2, center;
	pos1 = glm::vec3(d.part(0).x(), d.part(0).y(), 0.0);
	pos2 = glm::vec3(d.part(16).x(), d.part(16).y(), 0.0);
	center = (pos1 + pos2) * 0.5;

	//ofLogNotice("num verts : " + ofToString(verts.size()));

	float radius = glm::distance(pos1, pos2) / 2.0;
	size_t num_pts = 10;
	for (size_t i = 1; i < num_pts-1; i++)
	{
		float angle = PI / ((float)num_pts - 1) * i;
		glm::vec3 new_pos = glm::vec3(
			cos(angle) * radius + center.x, 
			
			sin(-angle) * radius + center.y,

			0.0
		);
		verts.push_back(new_pos);
	}

	//ofLogNotice("num verts : " + ofToString(verts.size()));


	mesh.addVertices(verts);

	std::vector<unsigned int> indices = { 
		0,17,36,36,17,18,37,18,19,38,
		19,20,39,20,21,39,21,27,27,28,
		39,29,39,28,39,29,30,30,1,40,
		1,0,36,1,41,40,2,1,30,2,
		31,49,50,32,33,3,49,48,4,48,
		59,5,59,58,6,58,57,8,7,57,
		49,31,50,50,31,32,27,22,42,22,
		23,42,24,43,42,25,44,43,26,45,
		44,26,16,45,46,45,16,15,47,46,
		27,42,28,42,29,28,30,29,42,30,
		42,47,30,15,14,34,52,51,35,52,
		34,52,35,53,35,14,13,53,13,12,
		54,12,11,55,11,10,56,10,9,57,
		9,8,9,57,56,10,56,55,11,55,
		54,12,54,53,13,53,35,51,33,34,
		14,35,30,47,15,30,16,15,46,44,
		25,26,43,24,25,42,23,24,57,7,
		6,58,6,5,59,5,4,48,4,3,
		33,51,50,49,3,2,30,31,2,36,
		41,1,40,39,30,20,39,38,19,38,
		37,18,37,36,0,67,17,67,66,17,
		17,66,18,18,66,65,65,19,18,19,
		65,64,20,19,64,64,21,20,21,64,
		27,27,64,63,22,27,63,63,23,22,
		23,63,24,24,63,62,62,25,24,25,
		62,61,61,26,25,26,61,60,26,60,
		16
	};

	// forehead indices
	//indices.push_back(0); indices.push_back(17); indices.push_back(67);
	//indices.push_back(17); indices.push_back(67); indices.push_back(66);
	//indices.push_back(17); indices.push_back(18); indices.push_back(66);
	//indices.push_back(18); indices.push_back(65); indices.push_back(66);
	//indices.push_back(18); indices.push_back(19); indices.push_back(65);
	//indices.push_back(19); indices.push_back(64); indices.push_back(65);
	mesh.addIndices(indices.data(), indices.size());


	std::vector<glm::vec2> t_coords = { 
		glm::vec2(0.000000, 230.978714),glm::vec2(8.202797, 277.388123),glm::vec2(12.758161, 323.817841),
		glm::vec2(28.760233, 372.349030),glm::vec2(54.848812, 411.945587),glm::vec2(95.251251, 443.935822),
		glm::vec2(141.887695, 476.302856),glm::vec2(198.896622, 500.510345),glm::vec2(255.992035, 512.000000),
		glm::vec2(313.103394, 500.510345),glm::vec2(370.112335, 476.302856),glm::vec2(416.748749, 443.935822),
		glm::vec2(457.151215, 411.945587),glm::vec2(483.239838, 372.349030),glm::vec2(499.241852, 323.817841),
		glm::vec2(503.797211, 277.388123),glm::vec2(512.000000, 230.978729),glm::vec2(67.622635, 165.738831),
		glm::vec2(97.213753, 150.183105),glm::vec2(132.475891, 145.232437),glm::vec2(169.602234, 147.016434),
		glm::vec2(202.717819, 154.502106),glm::vec2(309.282196, 154.502106),glm::vec2(342.397797, 147.016434),
		glm::vec2(379.524139, 145.232437),glm::vec2(414.786224, 150.183105),glm::vec2(444.377411, 165.738831),
		glm::vec2(255.791733, 199.971649),glm::vec2(255.997482, 225.746155),glm::vec2(255.795776, 253.484955),
		glm::vec2(256.368652, 277.725983),glm::vec2(215.528519, 324.049011),glm::vec2(239.851624, 330.269592),
		glm::vec2(256.127625, 332.184540),glm::vec2(278.261841, 327.282898),glm::vec2(298.727875, 318.555328),
		glm::vec2(92.551414, 209.043015),glm::vec2(123.845032, 196.293381),glm::vec2(164.247986, 195.820969),
		glm::vec2(196.671844, 211.563019),glm::vec2(161.510254, 218.718918),glm::vec2(122.117149, 217.811951),
		glm::vec2(315.328156, 211.563019),glm::vec2(347.752045, 195.820969),glm::vec2(388.154999, 196.293381),
		glm::vec2(419.448608, 209.043015),glm::vec2(389.882904, 217.811951),glm::vec2(350.489777, 218.718918),
		glm::vec2(167.983261, 382.497040),glm::vec2(195.702652, 358.197357),glm::vec2(225.060654, 346.739777),
		glm::vec2(256.212555, 347.522827),glm::vec2(288.481415, 346.942902),glm::vec2(316.427124, 357.853943),
		glm::vec2(348.185669, 387.013824),glm::vec2(324.063934, 399.149017),glm::vec2(292.006226, 410.581207),
		glm::vec2(258.338623, 416.046783),glm::vec2(222.390457, 411.362946),glm::vec2(192.927856, 398.118683),
		glm::vec2(496.561310, 184.794586),glm::vec2(452.107391, 98.327156),glm::vec2(384.000000, 34.148655),
		glm::vec2(300.453949, 0.000012),glm::vec2(211.546097, 0.000000),glm::vec2(127.999992, 34.148670),
		glm::vec2(59.892628, 98.327141),glm::vec2(15.438691, 184.794556)

	};
	mesh.addTexCoords(t_coords);
	return mesh;
}
void Gui2oneFaceDetector::cvRenderFacesLandmarks(cv::Mat _frame , std::vector<dlib::full_object_detection>& shapes)
{
	

	for (size_t i = 0; i < shapes.size(); i++)
	{
		dlib::full_object_detection& shape = shapes[i];
		cv::drawMarker(_frame, cv::Point(shape.part( 0).x(), shape.part( 0).y()), cv::Scalar(20, 255, 255, 255));
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


		image_points2.push_back(cv::Vec2d(d0.part( 0).x(), d0.part( 0).y()));
		image_points2.push_back(cv::Vec2d(d0.part( 4).x(), d0.part( 4).y()));
		image_points2.push_back(cv::Vec2d(d0.part(12).x(), d0.part(12).y()));
		image_points2.push_back(cv::Vec2d(d0.part(16).x(), d0.part(16).y()));
		image_points2.push_back(cv::Vec2d(d0.part(17).x(), d0.part(17).y()));
		image_points2.push_back(cv::Vec2d(d0.part(21).x(), d0.part(21).y()));
		image_points2.push_back(cv::Vec2d(d0.part(22).x(), d0.part(22).y()));
		image_points2.push_back(cv::Vec2d(d0.part(26).x(), d0.part(26).y()));
		image_points2.push_back(cv::Vec2d(d0.part(27).x(), d0.part(27).y()));
		image_points2.push_back(cv::Vec2d(d0.part(30).x(), d0.part(30).y()));
		image_points2.push_back(cv::Vec2d(d0.part(36).x(), d0.part(36).y()));
		image_points2.push_back(cv::Vec2d(d0.part(39).x(), d0.part(39).y()));
		image_points2.push_back(cv::Vec2d(d0.part(42).x(), d0.part(42).y()));
		image_points2.push_back(cv::Vec2d(d0.part(45).x(), d0.part(45).y()));
		image_points2.push_back(cv::Vec2d(d0.part(48).x(), d0.part(48).y()));
		image_points2.push_back(cv::Vec2d(d0.part(54).x(), d0.part(54).y()));




		float aspect_ratio = (float)proc_width / (float)proc_height;
		float aov = desired_aov;
		float focalLength = (float)proc_width * ofDegToRad(aov);
		float opticalCenterX = (float)proc_width / 2.0;
		float opticalCenterY = (float)proc_height / 2.0;
		

		// make camera matrix
		cv::Mat cam_mat = cv::Mat::eye(3, 3, CV_64F);
		cam_mat.at<double>(0, 0) = focalLength;
		cam_mat.at<double>(0, 1) = 0.0;
		cam_mat.at<double>(0, 2) = opticalCenterX;

		cam_mat.at<double>(1, 0) = 0.0;
		cam_mat.at<double>(1, 1) = focalLength;
		cam_mat.at<double>(1, 2) = opticalCenterY;

		cam_mat.at<double>(2, 0) = 0.0;
		cam_mat.at<double>(2, 1) = 0.0;
		cam_mat.at<double>(2, 2) = 1.0;

		cv::Mat1d projectionMat = cv::Mat::zeros(3, 3, CV_32F);


		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); // = { 0.0, 0.0, 0.0 };
		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);


		cv::Mat1d dist_coeffs = cv::Mat::zeros(5, 1, CV_32F);

		//// example coefficients found here :https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
		dist_coeffs(0, 0) =  4.1802327176423804e-001;
		dist_coeffs(1, 0) = -5.0715244063187526e-001;
		dist_coeffs(2, 0) = 0.0;
		dist_coeffs(3, 0) = 0.0;
		dist_coeffs(4, 0) = 5.7843597214487474e-001;


		//cv::solvePnP(object_points2, image_points2, cam_mat, cv::noArray(), rvec, tvec, false , cv::SOLVEPNP_EPNP);
		cv::solvePnP(object_points2, image_points2, cam_mat, dist_coeffs, rvec, tvec, false , cv::SOLVEPNP_UPNP);
		//cv::solvePnP(object_points2, image_points2, cam_mat, cv::noArray(), rvec, tvec, false , cv::SOLVEPNP_DLS);
		//cv::solvePnPRansac(object_points2, image_points2, cam_mat, cv::noArray(), rvec, tvec, false , 100, 8.0, 0.999999, cv::noArray(), cv::SOLVEPNP_UPNP );

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

	}

	return tr_vectors;
}

TransformVectors Gui2oneFaceDetector::estimateTransforms(const dlib::full_object_detection & detection, dlib::rectangle rectangle, cv::Mat & _frame, float desired_aov, bool draw_infos)
{
	//object_points2.clear();
	
	TransformVectors  tr_vector;


		const dlib::full_object_detection& d0 = detection;
		image_points2.clear();


		image_points2.push_back(cv::Vec2d(d0.part(0).x(), d0.part(0).y()));
		image_points2.push_back(cv::Vec2d(d0.part(4).x(), d0.part(4).y()));
		image_points2.push_back(cv::Vec2d(d0.part(12).x(), d0.part(12).y()));
		image_points2.push_back(cv::Vec2d(d0.part(16).x(), d0.part(16).y()));
		image_points2.push_back(cv::Vec2d(d0.part(17).x(), d0.part(17).y()));
		image_points2.push_back(cv::Vec2d(d0.part(21).x(), d0.part(21).y()));
		image_points2.push_back(cv::Vec2d(d0.part(22).x(), d0.part(22).y()));
		image_points2.push_back(cv::Vec2d(d0.part(26).x(), d0.part(26).y()));
		image_points2.push_back(cv::Vec2d(d0.part(27).x(), d0.part(27).y()));
		image_points2.push_back(cv::Vec2d(d0.part(30).x(), d0.part(30).y()));
		image_points2.push_back(cv::Vec2d(d0.part(36).x(), d0.part(36).y()));
		image_points2.push_back(cv::Vec2d(d0.part(39).x(), d0.part(39).y()));
		image_points2.push_back(cv::Vec2d(d0.part(42).x(), d0.part(42).y()));
		image_points2.push_back(cv::Vec2d(d0.part(45).x(), d0.part(45).y()));
		image_points2.push_back(cv::Vec2d(d0.part(48).x(), d0.part(48).y()));
		image_points2.push_back(cv::Vec2d(d0.part(54).x(), d0.part(54).y()));


		//image_points2.push_back(cv::Vec2d(d0.part (0).x() - rectangle.left(), d0.part( 0).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part( 4).x() - rectangle.left(), d0.part( 4).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(12).x() - rectangle.left(), d0.part(12).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(16).x() - rectangle.left(), d0.part(16).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(17).x() - rectangle.left(), d0.part(17).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(21).x() - rectangle.left(), d0.part(21).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(22).x() - rectangle.left(), d0.part(22).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(26).x() - rectangle.left(), d0.part(26).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(27).x() - rectangle.left(), d0.part(27).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(30).x() - rectangle.left(), d0.part(30).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(36).x() - rectangle.left(), d0.part(36).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(39).x() - rectangle.left(), d0.part(39).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(42).x() - rectangle.left(), d0.part(42).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(45).x() - rectangle.left(), d0.part(45).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(48).x() - rectangle.left(), d0.part(48).y() - rectangle.top()));
		//image_points2.push_back(cv::Vec2d(d0.part(54).x() - rectangle.left(), d0.part(54).y() - rectangle.top()));


		float aspect_ratio = (float)proc_width / (float)proc_height;
		float aov = desired_aov;
		float focalLength = (float)proc_width * ofDegToRad(aov);
		float opticalCenterX = (float)proc_width / 2.0;
		float opticalCenterY = (float)proc_height / 2.0;


		// make camera matrix
		cv::Mat cam_mat = cv::Mat::eye(3, 3, CV_64F);
		cam_mat.at<double>(0, 0) = focalLength;
		cam_mat.at<double>(0, 1) = 0.0;
		cam_mat.at<double>(0, 2) = opticalCenterX;

		cam_mat.at<double>(1, 0) = 0.0;
		cam_mat.at<double>(1, 1) = focalLength;
		cam_mat.at<double>(1, 2) = opticalCenterY;

		cam_mat.at<double>(2, 0) = 0.0;
		cam_mat.at<double>(2, 1) = 0.0;
		cam_mat.at<double>(2, 2) = 1.0;

		cv::Mat1d projectionMat = cv::Mat::zeros(3, 3, CV_32F);


		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32F); // = { 0.0, 0.0, 0.0 };
		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F);


		cv::Mat1d dist_coeffs = cv::Mat::zeros(5, 1, CV_32F);

		//// example coefficients found here :https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
		dist_coeffs(0, 0) = 4.1802327176423804e-001;
		dist_coeffs(1, 0) = -5.0715244063187526e-001;
		dist_coeffs(2, 0) = 0.0;
		dist_coeffs(3, 0) = 0.0;
		dist_coeffs(4, 0) = 5.7843597214487474e-001;


		//cv::solvePnP(object_points2, image_points2, cam_mat, cv::noArray(), rvec, tvec, false , cv::SOLVEPNP_EPNP);
		cv::solvePnP(object_points2, image_points2, cam_mat, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_UPNP);
		//cv::solvePnP(object_points2, image_points2, cam_mat, cv::noArray(), rvec, tvec, false , cv::SOLVEPNP_DLS);
		//cv::solvePnPRansac(object_points2, image_points2, cam_mat, cv::noArray(), rvec, tvec, false , 100, 8.0, 0.999999, cv::noArray(), cv::SOLVEPNP_UPNP );

		// Black magic: The x axis in the rotation vector needs to get flipped.
		double * r = rvec.ptr<double>(0);
		r[0] *= -1;
		r[1] *= -1;

		TransformVectors result;
		result.translates = ofVec3f(tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2));
		result.rotates = ofVec3f(rvec.at<double>(0, 0), rvec.at<double>(0, 1), rvec.at<double>(0, 2));

		

		//if (draw_infos)
		//{
		//	// display Data on Image
		//	float font_size = 0.5;
		//	int font_thickness = 1;
		//	cv::Scalar font_color(255, 255, 255);


		//	int font_face = CV_FONT_HERSHEY_SIMPLEX;

		//	char buff[255];
		//	sprintf(buff, "Face %d", det_id);
		//	cv::Size text_size = cv::getTextSize(buff, font_face, font_size, font_thickness, 0);

		//	cv::Point text_pos = cv::Point(rectangle.tl_corner().x(), rectangle.tl_corner().y());
		//	cv::Point text_end = cv::Point(text_pos.x + text_size.width, (text_pos.y - text_size.height));
		//	cv::rectangle(_frame, text_pos, text_end, cv::Scalar(0, 0, 0, 0), CV_FILLED);
		//	cv::putText(_frame, buff, text_pos, CV_FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness);

		//	char buff_rot[255];
		//	sprintf(buff_rot, "rotation: %.3f %.3f %.3f", rvec.at<double>(0, 0), rvec.at<double>(0, 1), rvec.at<double>(0, 2));
		//	cv::Size text_size_rot = cv::getTextSize(buff_rot, font_face, font_size, font_thickness, 0);

		//	cv::Point text_pos_rot = cv::Point(rectangles[det_id].tl_corner().x(), rectangles[det_id].tl_corner().y() + 20);
		//	cv::Point text_end_rot = cv::Point(text_pos_rot.x + text_size_rot.width, text_pos_rot.y - text_size_rot.height);
		//	cv::rectangle(_frame, text_pos_rot, text_end_rot, cv::Scalar(0, 0, 0, 0), CV_FILLED);
		//	cv::putText(_frame, buff_rot, text_pos_rot, CV_FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness);

		//	char buff_trans[255];
		//	sprintf(buff_trans, "translation: %.3f %.3f %.3f", tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2));
		//	cv::Size text_size_trans = cv::getTextSize(buff_trans, font_face, font_size, font_thickness, 0);

		//	cv::Point text_pos_trans = cv::Point(rectangles[det_id].tl_corner().x(), rectangles[det_id].tl_corner().y() + 50);
		//	cv::Point text_end_trans = cv::Point(text_pos_trans.x + text_size_trans.width, text_pos_trans.y - text_size_trans.height);
		//	cv::rectangle(_frame, text_pos_trans, text_end_trans, cv::Scalar(0, 0, 0, 0), CV_FILLED);
		//	cv::putText(_frame, buff_trans, text_pos_trans, CV_FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness);


		//}

	

	return result;
}
