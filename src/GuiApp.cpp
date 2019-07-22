
#include "GuiApp.h"

static std::string & ltrim(std::string & str)
{
	auto it2 = std::find_if(str.begin(), str.end(), [](char ch) { return !std::isspace<char>(ch, std::locale::classic()); });
	str.erase(str.begin(), it2);
	return str;
}

static std::string & rtrim(std::string & str)
{
	auto it1 = std::find_if(str.rbegin(), str.rend(), [](char ch) { return !std::isspace<char>(ch, std::locale::classic()); });
	str.erase(it1.base(), str.end());
	return str;
}

static std::string & trim(std::string & str)
{
	return ltrim(rtrim(str));
}

void GuiApp::setup()
{

	fileDialog.setup();

	ofAddListener(fileDialog.fileDialogEvent, this, &GuiApp::onFileDialogResponse);
	// tinyfd_notifyPopup("ofxThreadedFileDialog", "a non-blocking cross-platformfile dialog\nbased on http://tinyfiledialogs.sourceforge.net", "info");

	this->gui.setup();
	ImGui::GetIO().MouseDrawCursor = false;

	ofSetVerticalSync(false);

	plot_values_rx.reserve(plot_history);
	for (size_t i = 0; i < plot_history; i++) {
		plot_values_rx.emplace_back(0.0f);
	}

	plot_values_ry.reserve(plot_history);
	for (size_t i = 0; i < plot_history; i++) {
		plot_values_ry.emplace_back(0.0f);
	}

	plot_values_rz.reserve(plot_history);
	for (size_t i = 0; i < plot_history; i++) {
		plot_values_rz.emplace_back(0.0f);
	}


	ofDisableLighting();

}


void GuiApp::update()
{
	if (plot_values_rx.size() > plot_history) {

		int num = plot_values_rx.size() - plot_history;
		plot_values_rx.erase(plot_values_rx.end() - num, plot_values_rx.end());

		plot_values_ry.erase(plot_values_ry.end() - num, plot_values_ry.end());

		plot_values_rz.erase(plot_values_rz.end() - num, plot_values_rz.end());
	}
}

void GuiApp::draw()
{
	ofDisableLighting();
	gui.begin();
	{
		//auto main_settings = ofxImGui::Settings();
		ImGui::Begin("Main Params", false);



		if (ImGui::Button("Save To File")) {
			openSaveFileDialog();
		}
		if (ImGui::Button("Load From File")) {
			openLoadFileDialog();
		}

		ImGui::DragFloat("Z Offset", &z_offset, 0.1f, -500.0f, 500.0f);
		ImGui::DragFloat("Global Scale", &global_scale,0.1f, 0.2f,10.0f);
		ImGui::DragFloat("Angle ", &aov,0.1f, 0.1f,180.0f);
	
		ImGui::PushItemWidth(-1);
		ImGui::PlotLines("", plot_values_rx.data(), plot_values_rx.size(), 0, "rx", -180, 180, ImVec2(0.f,100.f));
		ImGui::PlotLines("", plot_values_ry.data(), plot_values_rx.size(), 0, "ry", -180, 180, ImVec2(0.f,100.f));
		ImGui::PlotLines("", plot_values_rz.data(), plot_values_rx.size(), 0, "rz", -180, 180, ImVec2(0.f,100.f));





		ImGui::End();
	
	}

	gui.end();


}

void GuiApp::exit()
{
	//ofExit();
	
}


bool GuiApp::saveToTextFile(std::string path)
{

	
	//ofFileDialogResult result = ofSystemLoadDialog("load params text file");

	//if (result.getPath() != "") {
	//	
	//}
	//else {
	//	printf("path is empy\n");
	//}

	ofstream file(path);
	if (file.fail()) {

		ofLogError("file creation FAIL for :" + path);
		return false;
	}
	

	string save_str;
	save_str  = "Z Offset = " + ofToString(z_offset) + "\n";
	save_str += "Global Scale = " + ofToString(global_scale) + "\n";
	save_str += "Angle = " + ofToString(aov) + "\n";

	file.write(save_str.c_str(), save_str.size());

	file.close();
	ofLogNotice("successfully saved data to file -- " + path);
	return true;
}

bool GuiApp::loadFromTextFile(std::string path)
{
	ifstream input_file(path);
	
	if (input_file.fail()) {

		ofLogError("file open FAIL for :" + path);		
		return false;
	}
	string line;
	while (std::getline(input_file, line)) {

		std::string delimiter = "=";
		std::string var_name = trim(line.substr(0, line.find(delimiter)));
		std::string str_value = trim(line.substr(line.find(delimiter)+1, line.size()));
		
		if       (var_name == "Z Offset") {

			z_offset = stof(str_value);

		}else if (var_name == "Global Scale") {

			global_scale = stof(str_value);

		}else if (var_name == "Angle") {

			aov = stof(str_value);
		}

	}

	ofLogNotice("successfully loaded data from file -- " + path);
	return true;
}

void GuiApp::openSaveFileDialog()
{
	fileDialog.saveFile("save file", "save file as", "param_save_01.txt");
}

void GuiApp::openLoadFileDialog()
{
	fileDialog.openFile("load file", "load saved Parameters");
}

void GuiApp::onFileDialogResponse(ofxThreadedFileDialogResponse & response)
{
	if (response.id == "save file") {

		ofLogNotice("response : "+ response.filepath);
		saveToTextFile(response.filepath);
	}
	else if (response.id == "load file") {
		loadFromTextFile(response.filepath);
	}
	
}


