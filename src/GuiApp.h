#pragma once

#include "ofMain.h"
#include "ofxImGui.h"
#include "gui2one_face_detector.h"
#include <stdio.h>
#include <fstream>

#include "ofxThreadedFileDialog.h"


class GuiApp : public ofBaseApp
{
public:

	void setup();
	void update();
	void draw();




	bool saveToTextFile(std::string path);
	bool loadFromTextFile(std::string path);

	void openSaveFileDialog();
	void openLoadFileDialog();
	void onFileDialogResponse(ofxThreadedFileDialogResponse &response);


	float z_offset = 235.0;
	float global_scale = 1.0;
	float aov = 60.0;

	std::vector<float> plot_values_tx;
	std::vector<float> plot_values_ty;
	std::vector<float> plot_values_tz;

	std::vector<float> plot_values_rx;
	std::vector<float> plot_values_ry;
	std::vector<float> plot_values_rz;

	int plot_history = 50;


	ofxImGui::Gui gui;


	
	
	ofxThreadedFileDialog fileDialog;
	//ofxThreadedFileDialogResponse response;

};

