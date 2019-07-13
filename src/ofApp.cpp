#include "ofApp.h"
static ofRectangle dlib_rect_to_of(dlib::rectangle _rect) {
	return ofRectangle(_rect.left(), _rect.top(), _rect.width(), _rect.height());
}
//--------------------------------------------------------------
void ofApp::setup(){

	w_width = 640;
	w_height = 360;

	gui.setup();

	gui.add(z_offset_slider.setup("Z offset", 375.0, -500.0, 500.0));
	gui.add(aov_slider.setup("AOV", 45.0, 0.01, 120.0));
	gui.add(object_scale_slider.setup("Object Scale", 3.0, 0.5, 10.0));


	grabber.initGrabber(w_width, w_height);

	//video_player.load("Bill & Melinda Gates Talk Taxing The Wealthy.mp4");	
	//video_player.load("'Brooklyn+Nine+Nine'+Cast+on+Being+Saved+by+NBC+_+Comic-Con+2018+_+TVLine.mp4");	
	//video_player.play();

	of_image.allocate(w_width, w_height, OF_IMAGE_COLOR);

	//test_mesh = importer.loadFile("data/pig_head.obj");
	test_mesh = importer.loadFile("data/face_mask_1.fbx");
	test_mesh.enableNormals();

	std::vector<glm::vec3> normals = test_mesh.getNormals();
	
	//ofLog(OF_LOG_NOTICE, "normal 0 is : " + ofToString(normals));
	

	light_1.setPointLight();
	light_1.enable();
	camera.setFarClip(30000.0);
	//camera.rotateAround(180.0, glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, 0.0));
	camera.setVFlip(true);

	

	
	
}

//--------------------------------------------------------------
void ofApp::update(){

	
	//video_player.update();

	grabber.update();
	if (grabber.isFrameNew())
	{

		cv::Mat frame = ofxCv::toCv(grabber);
		cv::Mat small = cv::Mat(360, 640, CV_8UC3);
		ofxCv::resize(frame, small);
		
		rectangles.clear();
		rectangles = face_detector.detectFaces(small);
		std::vector<dlib::full_object_detection> dets = face_detector.detectLandmarks(rectangles, small);
		tr_vectors = face_detector.estimateTransforms(dets, rectangles, small, aov_slider, false);

		face_detector.cvRenderFacesLandmarks(small, dets);
		//printf("matrices num = %d\n", matrices.size());
		ofxCv::toOf(small, of_image);
		of_image.update();

		
	}
}

//--------------------------------------------------------------
void ofApp::draw(){

	
	//grabber.draw(0, 0,500,500);
	//of_image.draw(0, 0, w_width, w_height);
	grabber.draw(0, 0, w_width, w_height);

	


	test_objects.clear();
	
	for(size_t i=0; i < tr_vectors.size(); i++)
	{

		
		TransformVectors tr = tr_vectors[i];
		

		
		//std::cout << tr.translates << std::endl;

		ofQuaternion quat( 
			tr.rotates.x / PI * 180.0, ofVec3f(1.0, 0.0, 0.0),			
			tr.rotates.y / PI * 180.0, ofVec3f(0.0, 1.0, 0.0),
			tr.rotates.z / PI * 180.0, ofVec3f(0.0, 0.0, 1.0)
		);
		

		//box.setGlobalOrientation(orientation);
		//box.rotate(orientation);
		
		MeshObject obj;
		obj.setMesh(&test_mesh);
		
		obj.setScale(1000.0* object_scale_slider);
		obj.setPosition(ofVec3f((tr.translates.x  * w_width ) + w_width / 2.0 , (tr.translates.y  * w_height/(1.0/(float(w_width)/ w_height))) + w_height / 2.0, (-tr.translates.z) * z_offset_slider));
		obj.setOrientation(quat);

		

		test_objects.push_back(obj);
		

	}
	
	ofEnableDepthTest();
	light_1.enable();
	camera.begin();
	for (size_t i = 0; i < test_objects.size(); i++)
	{
		ofPushMatrix();
		
		test_objects[i].draw();

		ofPopMatrix();
	}

	camera.end();
	light_1.disable();
	ofDisableDepthTest();

	ofNoFill();
	for (size_t rect_id = 0; rect_id < rectangles.size(); rect_id++)
	{
		ofDrawRectangle(dlib_rect_to_of(rectangles[rect_id]));
	}
	ofFill();
	gui.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
	w_width = w;
	w_height = h;
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
