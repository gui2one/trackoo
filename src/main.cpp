#include "ofMain.h"
#include "ofApp.h"

//========================================================================
int main(int argc, char* argv[]) {
	ofSetupOpenGL(640,360,OF_WINDOW);			// <-------- setup the GL context

	std::string video_file_path = "";
	if (argc < 2) {

		printf("no arguments given\n");
	}
	else {

		printf("argument : %s \n", argv[1]);
		printf("value : %s \n", argv[2]);

		std::string arg_str = argv[1];
		if (arg_str == "-v") {
			video_file_path = argv[2];
			printf("path : %s\n", video_file_path.c_str());
		}
	}
	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN
	// pass in width and height too:
	ofApp * myApp = new ofApp();

	myApp->video_file_path = video_file_path;
	ofRunApp(myApp);

}
