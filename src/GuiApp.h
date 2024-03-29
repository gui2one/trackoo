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

	void exit();




	bool saveToTextFile(std::string path);
	bool loadFromTextFile(std::string path);

	void openSaveFileDialog();
	void openLoadFileDialog();
	void onFileDialogResponse(ofxThreadedFileDialogResponse &response);


	shared_ptr<ofAppGLFWWindow> base_window;
	//shared_ptr<ofBaseGLRenderer> gl_renderer;

	float z_offset = 235.0;
	float global_scale = 1.0;
	float aov = 60.0;

	bool b_show_rectangles = false;

	bool b_show_polylines = false;
	bool b_show_3d_heads = false;
	bool b_show_polymasks = false;
	bool b_show_player = true;
	bool b_show_hats = false;

	int current_player_texture = 0;

	float player_heads_scale_offset = 0.4f;

	bool b_show_overlay = true;
	bool b_proc_width_changed = false;
	int proc_width = 520;

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

