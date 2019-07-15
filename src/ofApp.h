#pragma once

#include "ofMain.h"
#include "gui2one_face_detector.h"
#include "ofxCv.h"
#include "ofxGui.h"


//#include "assimp/Importer.hpp"
//#include "assimp/scene.h"
//#include "assimp/postprocess.h"

#include "ObjectImporter.h"
#include "MeshObject.h"

//#include "glm/gtx/matrix_decompose.hpp"
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


		Gui2oneFaceDetector face_detector;

		ofVideoGrabber grabber;
		ofVideoPlayer video_player;
		ofImage of_image;

		int w_width , w_height;
		std::vector<ofMatrix4x4> matrices;
		std::vector<TransformVectors> tr_vectors;
		
		std::vector<dlib::rectangle> rectangles;
		


		ofxPanel gui;
		ofxFloatSlider z_offset_slider;
		ofxFloatSlider aov_slider;
		ofxFloatSlider object_scale_slider;

		ObjectImporter importer;

		ofMesh test_mesh;
		std::vector<ofMesh> test_meshes;
		std::vector<MeshObject> test_objects;

		ofCamera camera;
		std::string video_file_path;

		ofLight light_1;

		
		
};
