#include "ofApp.h"
static ofRectangle dlib_rect_to_of(dlib::rectangle _rect) {
	return ofRectangle(_rect.left(), _rect.top(), _rect.width(), _rect.height());
}

static std::vector<ofRectangle> dlib_rects_to_of(std::vector<dlib::rectangle> _rects) {
	std::vector<ofRectangle> result;
	for (auto rect : _rects) {
		ofRectangle of_rect( rect.left(), rect.top(), rect.width(), rect.height());
		result.push_back(of_rect);
	}
	return result;
}

static std::vector<dlib::rectangle> cv_rects_to_dlib(std::vector<cv::Rect> _rects) {

	std::vector<dlib::rectangle> dlib_rects;

	for (auto cv_rect : _rects) {
		dlib::rectangle dlib_rect(cv_rect.x, cv_rect.y, cv_rect.width, cv_rect.height);

		dlib_rects.push_back(dlib_rect);
	}

	return dlib_rects;

}

static std::vector<cv::Rect> dlib_rects_to_cv(std::vector<dlib::rectangle> _rects) {
	std::vector<cv::Rect> result;
	for (auto rect : _rects) {
		cv::Rect cv_rect(rect.left(), rect.top(), rect.width(), rect.height());
		result.push_back(cv_rect);
	}
	return result;
}


static std::vector<gui2oneFaceDetectorInstance> dlib_rects_to_instance(std::vector<dlib::rectangle> _rects) {
	
	std::vector<gui2oneFaceDetectorInstance> result;
	for (auto rect : _rects) {
		gui2oneFaceDetectorInstance instance;
		instance.x = rect.left();
		instance.y = rect.top();
		instance.width = rect.width();
		instance.height = rect.height();
		result.push_back(instance);
	}
	return result;
}



//
//void ofApp::windowCloseCallBack(GLFWwindow * _window) {
//
//	if (glfwGetWindowUserPointer(_window) != NULL) {
//
//		ofApp * user_ptr = static_cast<ofApp*>(glfwGetWindowUserPointer(_window));
//		std::cout << "was not NULL " << user_ptr->proc_width <<std::endl;
//	}
//	else {
//		std::cout << "was NULL " << std::endl;
//	}
//
//	glfwSetWindowShouldClose(_window, false);
//	
//}

//--------------------------------------------------------------
void ofApp::setup(){


	

	//glfwSetWindowUserPointer(glfw_window, (void *)this);
	//glfwSetWindowCloseCallback(glfw_window, windowCloseCallBack);

	

	

	w_width = 640;
	w_height = 360;

	proc_width = 520;
	proc_height = (int)((float)proc_width / (16.0/9.0));

	face_detector.setProcessSize(proc_width, proc_height);


	//grabber.initGrabber(w_width, w_height);
	
	video_player.load(video_file_path);
	video_player.play();

	of_image.allocate(w_width, w_height, OF_IMAGE_COLOR);
	

	
	test_mesh = importer.loadFile("data/face_mask_1.fbx");
	test_mesh.enableNormals();
	test_mesh.enableTextures();

	gl->enableLighting();
	gl->enableLight(0);
	light_1.setPointLight();
	light_1.enable();
	camera.setFarClip(30000.0);
	//camera.rotateAround(180.0, glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, 0.0));
	camera.setVFlip(true);
	
	ofLoadImage(texture, "face_texture_1.png");
	
	texture.setTextureWrap(GL_REPEAT, GL_REPEAT);
	//texture.setTextureMinMagFilter(GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
	
	
	std::vector<glm::vec2> scaled_coords;
	scaled_coords.reserve(test_mesh.getNumTexCoords());
	for (auto in_coord : test_mesh.getTexCoords()) {
		glm::vec2 coord;
		coord.x = in_coord.x * 512;
		coord.y = in_coord.y * 512.0;

		scaled_coords.emplace_back(coord);
	}

	test_mesh.clearTexCoords();
	test_mesh.addTexCoords(scaled_coords);

	rect_tracker.setMaximumDistance(200.0);
	rect_tracker.setPersistence(10);
	

	std::cout << "cuda devices : " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
	//im_gui.setup();

	//ImGui::GetIO().MouseDrawCursor = false;

	createGuiWindow();
	
}



//--------------------------------------------------------------
void ofApp::update(){

	

	
	//video_player.update();

	video_player.update();
	if (video_player.isFrameNew())
	{

		cv::Mat frame = ofxCv::toCv(video_player);
		cv::Mat small = cv::Mat(proc_height, proc_width, CV_8UC3);
		ofxCv::resize(frame, small);
		
		rectangles.clear();
		rectangles = face_detector.detectFaces(small);
		rect_tracker.track(dlib_rects_to_cv(rectangles));

		tracker_follower.track(dlib_rects_to_instance(rectangles));
		
		std::vector<dlib::rectangle> current_rects;
		for (auto label : rect_tracker.getCurrentLabels()) {

			auto cv_rect = rect_tracker.getCurrent(label);
			auto dlib_rect = dlib::rectangle(cv_rect.x, cv_rect.y, cv_rect.x + cv_rect.width, cv_rect.y + cv_rect.height);

			current_rects.push_back(dlib_rect);
		}
		std::vector<dlib::full_object_detection> dets = face_detector.detectLandmarks(current_rects, small);
		tr_vectors = face_detector.estimateTransforms(dets, rectangles, small, im_gui->aov, false);

		face_detector.cvRenderFacesLandmarks(small, dets);
		
		//ofxCv::toOf(small, of_image);
		//of_image.update();

		
	}


	
}

//--------------------------------------------------------------
void ofApp::draw(){



	ofDisableLighting();

	gl->draw(video_player, 0, 0, w_width, w_height);
	


	test_objects.clear();
	
	for(size_t i=0; i < tr_vectors.size(); i++)
	{
		
		TransformVectors tr = tr_vectors[i];

		ofQuaternion quat( 
			ofRadToDeg(tr.rotates.x), ofVec3f(1.0, 0.0, 0.0),
			ofRadToDeg(tr.rotates.y), ofVec3f(0.0, 1.0, 0.0),
			ofRadToDeg(tr.rotates.z), ofVec3f(0.0, 0.0, 1.0)
		);
		

		MeshObject obj;
		obj.renderer = gl;
		obj.setMesh(&test_mesh);
		
		obj.setScale(1000.0* im_gui->global_scale);
		obj.setOrientation(quat);
		obj.setPosition(ofVec3f((tr.translates.x  * proc_width ) + proc_width / 2.0 , (tr.translates.y  * proc_height/(1.0/(float(proc_width)/ proc_height))) + proc_height / 2.0, (-tr.translates.z) * im_gui->z_offset));

		

		test_objects.push_back(obj);

	}
	
	if (tr_vectors.size() > 0) {

		im_gui->plot_values_rx.insert(im_gui->plot_values_rx.begin(), tr_vectors[0].rotates.x / PI  * 180.0);
		im_gui->plot_values_ry.insert(im_gui->plot_values_ry.begin(), tr_vectors[0].rotates.y / PI  * 180.0);
		im_gui->plot_values_rz.insert(im_gui->plot_values_rz.begin(), tr_vectors[0].rotates.z / PI  * 180.0);
		
	}
	

	ofEnableDepthTest();
	//ofDisableArbTex();
	light_1.enable();
	
	
	
	camera.begin();
	for (size_t i = 0; i < test_objects.size(); i++)
	{
		//ofPushMatrix();
		gl->setColor(255, 255, 255);
		//gl->enableTextureTarget(texture, 0);
		ofDisableArbTex();
		gl->bind(texture, 0);
		gl->draw(test_objects[i]);
		//gl->disableTextureTarget(0, 0);
		gl->unbind(texture, 0);
		//ofPopMatrix();
	}

	camera.end();
	
	
	//light_1.disable();

	ofDisableDepthTest();
	//ofDisableLighting();
	
	gl->setFillMode(OF_OUTLINE);

	
	float width_ratio = (float)w_width / (float)proc_width;
	float height_ratio = (float)w_height / (float)proc_height;
	for (auto label : rect_tracker.getCurrentLabels()) {

		auto cv_rect = rect_tracker.getCurrent(label);
		gl->setColor(255, 30, 30);
		gl->drawRectangle(cv_rect.x * width_ratio, cv_rect.y * height_ratio, 0.0, cv_rect.width * width_ratio,	 cv_rect.height * height_ratio);
		
		gl->setColor(255, 255, 255);
		gl->drawString("face id : " + ofToString(label), cv_rect.x * width_ratio, cv_rect.y * height_ratio, 0.0);

	}

	std::vector<my_type> followers = rect_tracker.getFollowers();

	for (size_t i = 0; i < followers.size(); i++) {

		int _label = followers[i].getLabel();
		if (rect_tracker.existsCurrent(_label)) {

			cv::Rect _rect = rect_tracker.getCurrent(_label);
			gl->setColor(0, 255, 0);
			gl->drawRectangle(_rect.x, _rect.y, 0, _rect.width, _rect.height);
		}
		else {
			//gl->setColor(255, 0, 0);
		}
		followers[i].my_setup();


	}

	gl->setColor(255, 255, 255);
	gl->setFillMode(OF_FILLED);

	gl->drawString(ofToString(ofGetFrameRate()), 10,30,0.0);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
	if (key == OF_KEY_RETURN) {
		printf(" enter pressed\n");
	}
	if (key == OF_KEY_RETURN && exiting == true) {

		ofExit();
	}
	if (key != 'q') {
		exiting = false;
	}
	if (key == 'f') {
		
		ofToggleFullscreen();
	}
	else if (key == 'q') {

		exiting = true;
		//// need to confirm exit by pressing Enter Key
	}
	else if (key == 'i') {
		if (gui_closed) {

			createGuiWindow();
			gui_closed = false;

		}
	}
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

void ofApp::exit()
{
	
	//return false;
	ofExit();

}

bool ofApp::my_exit() {
	return false;
}

bool ofApp::createGuiWindow()
{

	ofGLFWWindowSettings settings;

	settings.setSize(400, 400);
	settings.setPosition(ofVec2f(-800, 100));
	settings.resizable = true;
	settings.windowMode = OF_WINDOW;
	//settings.shareContextWith = nullptr;
	//settings.decorated = false;
	
	shared_ptr<ofAppGLFWWindow>  gui_window = static_pointer_cast<ofAppGLFWWindow>(ofCreateWindow(settings));

	// maxmize window 
	GLFWwindow * glfw_gui_window = gui_window->getGLFWWindow();
	glfwMaximizeWindow(glfw_gui_window);
	
	shared_ptr<GuiApp> guiApp(new GuiApp);

	im_gui = guiApp;
	im_gui->base_window = gui_window;
	//im_gui->gl_renderer = dynamic_pointer_cast<ofBaseGLRenderer>(gui_window->renderer());
	ofAddListener(im_gui->base_window->events().exit, this, &ofApp::onGuiExit);

	ofRunApp(im_gui->base_window, im_gui);
	
	
	return true;
}

void ofApp::onGuiExit(ofEventArgs& args ) 
{

	
	ofLogNotice("ofApp.cpp", "GUI EXIT !!!!!!!");
	ofRemoveListener(im_gui->base_window->events().exit, this, &ofApp::onGuiExit);
	gui_closed = true;
	
	//createGuiWindow();
}


//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
