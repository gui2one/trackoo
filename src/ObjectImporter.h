#pragma once
#include "ofMain.h"

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"


class ObjectImporter {
public : 
	//ObjectImporter();
	//~ObjectImporter();

	ofMesh loadFile(std::string file);
private:

	Assimp::Importer importer;
};