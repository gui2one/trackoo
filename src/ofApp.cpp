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





//--------------------------------------------------------------
void ofApp::setup(){
	
	
	w_width = 1920;
	w_height = 1080;




	//grabber.initGrabber(w_width, w_height);
	
	video_player.load(video_file_path);
	video_player.play();


	

	
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
	
	ofLoadImage(texture_logo, "logo_SRFC_01.png");
	ofLoadImage(texture_2_couleurs, "couleurs_SRFC_01.png");
	
	texture_logo.setTextureWrap(GL_REPEAT, GL_REPEAT);
	texture_logo.setTextureMinMagFilter(GL_LINEAR, GL_LINEAR);
	texture_logo.setTextureMinMagFilter(GL_NEAREST, GL_LINEAR);
	

	ofLoadImage(overlay_1, "overlay_01.png");

	ofTexture player_texture_9;
	ofLoadImage(player_texture_9, "joueurs/9_JORDAN_SIEBATCHEU.png");
	player_textures.push_back(player_texture_9);

	ofTexture player_texture_11;
	ofLoadImage(player_texture_11, "joueurs/11_M_BAYE_NIANG.png");
	player_textures.push_back(player_texture_11);

	ofTexture player_texture_14;
	ofLoadImage(player_texture_14, "joueurs/14_BENJAMIN_BOURIGEAUD.png");
	player_textures.push_back(player_texture_14);

	ofTexture player_texture_17;
	ofLoadImage(player_texture_17, "joueurs/17_FAITOUT_MAOUASSA.png");
	player_textures.push_back(player_texture_17);
	




	ofTexture hat_texture_1;
	ofLoadImage(hat_texture_1, "hats/HAT_3.png");
	hat_textures.push_back(hat_texture_1);



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
	

	std::cout << "OpenCV cuda devices : " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
	//im_gui.setup();

	//ImGui::GetIO().MouseDrawCursor = false;
	createGuiWindow();

	proc_width = im_gui->proc_width;
	proc_height = (int)((float)proc_width / ((float)w_width / (float)w_height));

	face_detector.setProcessSize(proc_width, proc_height);
	
}



//--------------------------------------------------------------
void ofApp::update(){


	if (im_gui->b_proc_width_changed) {

		proc_width = im_gui->proc_width;
		proc_height = (int)((float)proc_width / ((float)w_width / (float)w_height));
		face_detector.setProcessSize(proc_width, proc_height);
		im_gui->b_proc_width_changed = false;
		face_detector.initCvDnnNet();
	}
	video_player.update();

	if (video_player.isFrameNew())
	{

		cv::Mat frame = ofxCv::toCv(video_player);
		
		cv::Mat small = cv::Mat(proc_height, proc_width, CV_8UC3);

		
		ofxCv::resize(frame, small);
		cv::cuda::GpuMat gpu_mat;
		gpu_mat.upload(small);
		
		
		rectangles.clear();
		rectangles = face_detector.detectFaces(small);
		rect_tracker.track(dlib_rects_to_cv(rectangles));

		instance_tracker.track(dlib_rects_to_instance(rectangles));
		
		//std::vector<dlib::rectangle> current_rects;
	
		instances.clear();

		all_polylines.clear();
		all_poly_masks.clear();

		for (auto label : instance_tracker.getCurrentLabels()) {

			auto instance = instance_tracker.getCurrent(label);
			
			auto dlib_rect = dlib::rectangle(instance.x, instance.y, instance.x + instance.width, instance.y + instance.height);

			dlib::full_object_detection detection = face_detector.detectLandmarks(dlib_rect, small);


			if (im_gui->b_show_polylines) {

				std::vector<ofPolyline> polylines = face_detector.getPolylines(detection);
				all_polylines.push_back(polylines);
			}

			if (im_gui->b_show_polymasks) {

				ofMesh mask_mesh = face_detector.getMesh(detection);
				all_poly_masks.push_back(mask_mesh);
			}
			if (im_gui->b_show_3d_heads) {
				TransformVectors vectors = face_detector.estimateTransforms(detection, dlib_rect, small, im_gui->aov, false);
				instance.vectors = vectors;
				instances.push_back(instance);
			}
			
		}



		
	}


	
}

//--------------------------------------------------------------
void ofApp::draw(){
	float width_ratio = (float)w_width / (float)proc_width;
	float height_ratio = (float)w_height / (float)proc_height;
	ofEnableBlendMode(OF_BLENDMODE_ALPHA);

	ofDisableLighting();

	gl->draw(video_player, 0, 0, w_width, w_height);
	
	test_objects.clear();
	
	tr_vectors.clear();

	//for (auto instance : instances) {
	//	ofLogNotice(ofToString(instance.vectors.translates));
	//}


	for (auto polylines : all_polylines) {
		for (auto polyline : polylines) {

			gl->pushMatrix();
			float scale_factor = (float)w_width / (float)proc_width;

			float w_ratio = (float)w_width / (float)w_height;
			float proc_ratio = (float)proc_width / (float)proc_height;
			gl->scale(scale_factor, scale_factor * 1.0/( w_ratio / proc_ratio), scale_factor);
			gl->setLineWidth(2);
			gl->draw(polyline);

			gl->popMatrix();
		}
	}

	texture_2_couleurs.bind();
	gl->pushMatrix();
	float scale_factor = (float)w_width / (float)proc_width;

	float w_ratio = (float)w_width / (float)w_height;
	float proc_ratio = (float)proc_width / (float)proc_height;
	gl->scale(scale_factor, scale_factor * 1.0 / (w_ratio / proc_ratio), scale_factor);

	ofEnableBlendMode(OF_BLENDMODE_MULTIPLY);
	for (auto mask : all_poly_masks) {
		
		gl->draw(mask, OF_MESH_FILL);
	}

	ofEnableBlendMode(OF_BLENDMODE_ALPHA);
	texture_2_couleurs.unbind();
	gl->popMatrix();

	for(size_t i=0; i < instances.size(); i++)
	{
		
		TransformVectors tr = instances[i].vectors;

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
	
	if (instances.size() > 0) {

		im_gui->plot_values_rx.insert(im_gui->plot_values_rx.begin(), instances[0].vectors.rotates.x / PI  * 180.0);
		im_gui->plot_values_ry.insert(im_gui->plot_values_ry.begin(), instances[0].vectors.rotates.y / PI  * 180.0);
		im_gui->plot_values_rz.insert(im_gui->plot_values_rz.begin(), instances[0].vectors.rotates.z / PI  * 180.0);
		
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
		gl->bind(texture_logo, 0);
		gl->draw(test_objects[i]);
		//gl->disableTextureTarget(0, 0);
		gl->unbind(texture_logo, 0);
		//ofPopMatrix();
	}

	camera.end();
	
	//light_1.disable();


	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	ofEnableDepthTest();
	if (im_gui->b_show_player) {

		int inc = 0;
		float offset = im_gui->player_heads_scale_offset;
		for (auto follower : rect_tracker.getFollowers()) {

			int label = follower.getLabel();
			auto rect = rect_tracker.getCurrent(label);
			ofTexture& tex = player_textures[im_gui->current_player_texture];
			ofVec2f center;

			float size_ratio_2 =   tex.getHeight() / rect.height;
			center = ofVec2f(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
			//gl->draw(tex, 0.0f, 0.0f, 0.0f, (float)tex.getWidth(), (float)tex.getHeight(), 1.0f, 1.0f, (float)tex.getWidth(), (float)tex.getHeight());
			gl->draw(
				tex, // texture 
				(rect.x - rect.width * offset) * width_ratio , // start X
				(rect.y - rect.height * offset)* height_ratio, // start Y
				(float)inc * 0.1f, // start Z ?
				(rect.width   + (rect.width  * offset) * 2.0) * width_ratio, // end X
				(rect.height  + (rect.height * offset) * 2.0) * height_ratio, // end Y
				1.0f, 1.0f, 
				(float)tex.getWidth(), (float)tex.getHeight());// texture size

			inc++;
		}
	}

	if(im_gui->b_show_hats) {
		int inc = 0;
		float offset = im_gui->player_heads_scale_offset;
		for (auto follower : rect_tracker.getFollowers()) {

			int label = follower.getLabel();
			auto rect = rect_tracker.getCurrent(label);
			ofTexture& tex = hat_textures[0];
			ofVec2f center;

			float size_ratio_2 = tex.getHeight() / rect.height;
			center = ofVec2f(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
			//gl->draw(tex, 0.0f, 0.0f, 0.0f, (float)tex.getWidth(), (float)tex.getHeight(), 1.0f, 1.0f, (float)tex.getWidth(), (float)tex.getHeight());
			gl->draw(
				tex, // texture 
				(rect.x - rect.width * offset) * width_ratio, // start X
				((rect.y - rect.height * offset) - rect.height*0.1)* height_ratio, // start Y
				(float)inc * 0.1f, // start Z ?
				(rect.width + (rect.width  * offset) * 2.0) * width_ratio, // end X
				(rect.height + (rect.height * offset) * 2.0) * height_ratio * 0.5, // end Y
				1.0f, 1.0f,
				(float)tex.getWidth(), (float)tex.getHeight());// texture size

			inc++;
		}

	}
	//overlay_1.draw(0, 0, w_width, w_height);


	ofDisableDepthTest();
	//ofDisableLighting();
	if (im_gui->b_show_rectangles) {


		gl->setFillMode(OF_OUTLINE);



		for (auto label : rect_tracker.getCurrentLabels()) {

			auto cv_rect = rect_tracker.getCurrent(label);
			gl->setColor(255, 30, 30);
			gl->drawRectangle(cv_rect.x * width_ratio, cv_rect.y * height_ratio, 0.0, cv_rect.width * width_ratio, cv_rect.height * height_ratio);

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
	}



	ofDisableLighting();
	ofDisableDepthTest();

	if (im_gui->b_show_overlay) 
	{
		gl->draw(overlay_1, 0.0f, 0.0f, 0.0f, (float)w_width, (float)w_height, 1.0f, 1.0f, (float)overlay_1.getWidth(), (float)overlay_1.getHeight());
	}
		
	gl->drawString(ofToString(ofGetFrameRate()), 10,30,0.0);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
	if (key == OF_KEY_RETURN) {
		//printf(" enter pressed\n");
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
	//proc_width = 1024;
	//proc_height = (int)((float)proc_width / (16.0/9.0));

	face_detector.setProcessSize(proc_width, proc_height);
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
	settings.setPosition(ofVec2f(500, 100));
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
