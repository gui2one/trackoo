#include "ofMain.h"
#include "GuiApp.h"
#include "ofApp.h"
#include "ofAppGLFWWindow.h"


#define DLIB_USE_CUDA




static void windowCloseCallBkack(GLFWwindow * _window) {

	//ofApp * user_ptr = static_cast<ofApp*>(glfwGetWindowUserPointer(_window));
	//std::cout << "closing " << user_ptr->proc_width <<std::endl;

	std::cout << "Confirm Closing " << std::endl;
	glfwSetWindowShouldClose(_window, false);

}
//========================================================================
int main(int argc, char* argv[]) {
	//ofSetupOpenGL(640,360,OF_WINDOW);			// <-------- setup the GL context

	ofSetLogLevel(OF_LOG_VERBOSE);

	// prevent quit on press escape key
	ofSetEscapeQuitsApp(false);

	std::string video_file_path = "";
	if (argc < 2) {

		printf("no arguments given ... \nI need some aaarggss man ...\n");
	}
	else {

		//printf("argument : %s \n", argv[1]);
		//printf("value : %s \n", argv[2]);

		std::string arg_str = argv[1];
		if (arg_str == "-v") {
			video_file_path = argv[2];
			//printf("path : %s\n", video_file_path.c_str());
		}
	}


	ofGLFWWindowSettings settings;

	settings.setSize(640, 360);
	settings.setPosition(ofVec2f(500,100));
	settings.resizable = false;
	settings.decorated = true;
	shared_ptr<ofAppGLFWWindow> main_window = static_pointer_cast<ofAppGLFWWindow>(ofCreateWindow(settings));
	GLFWwindow * glfw_main = main_window->getGLFWWindow();

	//settings.setSize(400, 400);
	//settings.setPosition(ofVec2f(0, 100));
	//settings.resizable = true;
	//settings.decorated = false;
	//shared_ptr<ofAppBaseWindow> gui_window = ofCreateWindow(settings);

	shared_ptr<ofApp> mainApp(new ofApp);
	//shared_ptr<GuiApp> guiApp(new GuiApp);


	//mainApp->im_gui = guiApp;
	
	
	mainApp->glfw_window = glfw_main;
	mainApp->video_file_path = video_file_path;
	mainApp->gl = dynamic_pointer_cast<ofBaseGLRenderer>(main_window->renderer());
	
	//ofRunApp(gui_window, guiApp);
	ofRunApp(main_window, mainApp);

	
	

	
	glfwSetWindowCloseCallback(glfw_main, windowCloseCallBkack);


	ofRunMainLoop();
	

}
