#pragma once


#include "ofMain.h"
#include "gui2one_face_detector.h"
#include "ofxCv.h"

#include "GuiApp.h"

#include "ObjectImporter.h"
#include "MeshObject.h"


//#include <opencv2/opencv.hpp>
//#include <opencv2/cudaimgproc.hpp>

class my_type : public ofxCv::RectFollower, TransformVectors {

public:

	
	void my_setup() {
		
		//std::cout << "follower setup " <<  getLabel() << std::endl;
		

	}

private:

};


class ofApp : public ofBaseApp{

	public:
		void setup();
		
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		void exit();

		bool my_exit();


		bool exiting = false;
		bool confirm_exit = false;


		bool createGuiWindow();

		void onGuiExit(ofEventArgs& args);

		//static void windowCloseCallBack(GLFWwindow * _window);




		
		

		GLFWwindow * glfw_window;

		shared_ptr<ofBaseGLRenderer> gl;
		shared_ptr<GuiApp> im_gui = nullptr;
		bool gui_closed = false;
		

		Gui2oneFaceDetector face_detector;

		ofVideoGrabber grabber;
		ofVideoPlayer video_player;

		ofTexture overlay_1;
		ofImage overlay_img;
		//ofImage of_image;

		int w_width , w_height, proc_width, proc_height;

		std::vector<ofMatrix4x4> matrices;
		std::vector<TransformVectors> tr_vectors;
		

		//ofxCv::Tracker<cv::Rect> rect_tracker;
		ofxCv::TrackerFollower< cv::Rect, my_type> rect_tracker;
		ofxCv::TrackerFollower<gui2oneFaceDetectorInstance, MyFollower> instance_tracker;
		std::vector<gui2oneFaceDetectorInstance> instances;


		std::vector< std::vector<ofPolyline> > all_polylines;
		std::vector<ofMesh> all_poly_masks;

		std::vector<dlib::rectangle> rectangles;
		std::vector<dlib::rectangle> current_rectangles;


		ObjectImporter importer;

		ofMesh test_mesh;
		ofTexture texture_logo, texture_2_couleurs;
		std::vector<ofMesh> test_meshes;
		std::vector<MeshObject> test_objects;
		

		ofCamera camera;
		std::string video_file_path;

		ofLight light_1;

		std::vector<ofTexture> player_textures;
		

		

		
};
