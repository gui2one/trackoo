
#include "GuiApp.h"


void GuiApp::setup()
{
	gui.setup();
	ImGui::GetIO().MouseDrawCursor = false;

	ofSetVerticalSync(false);

	plot_values.reserve(plot_history);
	for (size_t i = 0; i < plot_history; i++) {
		plot_values.emplace_back(0.0f);
	}
}

void GuiApp::update()
{
	if (plot_values.size() > plot_history) {

		int num = plot_values.size() - plot_history;
		plot_values.erase(plot_values.end() - num, plot_values.end());
	}
}

void GuiApp::draw()
{

	gui.begin();

	ImGui::Begin("Main Params");
	
	ImGui::DragFloat("Z Offset", &z_offset, 0.1f, -500.0f, 500.0f);
	ImGui::DragFloat("Global Scale", &global_scale,0.1f, 0.2f,10.0f);
	ImGui::DragFloat("Angle ", &aov,0.1f, 0.1f,180.0f);
	



	ImGui::PlotLines("my plot", plot_values.data(), plot_values.size(), 0, "plot", -180, 180, ImVec2(200.f,300.f));
	
	ImGui::End();

	gui.end();
}
