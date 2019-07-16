#pragma once

#include "ofMain.h"
#include "ofxImGui.h"
class GuiApp : public ofBaseApp
{
public:
	void setup();
	void update();
	void draw();

	ofxImGui::Gui gui;

	float z_offset = 200.0;
	float global_scale = 1.0;

	std::vector<float> plot_values;
	int plot_history = 50;
};

