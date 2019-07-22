#include "ObjectImporter.h"
#include "ofMain.h"

ofMesh ObjectImporter::loadFile(std::string file)
{
	ofMesh mesh;
	const aiScene * scene = importer.ReadFile(file,
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType |
		aiProcess_GenNormals 
	);


	if (!scene) {
		ofLogError("ObjectImporter", "issue with Assimp !");
		//std::cout << "issue with Assimp !\n" << std::endl;
	}
	else {

		if (!scene->HasMeshes())
		{
			ofLogNotice("ObjectImporter", "Object is empty !");
			//std::cout << "Object is empty !\n" << std::endl;
		}
		else {

			//std::cout << "Custom Assimp loader GOOD !\n" << std::endl;
			
			for (size_t mesh_id = 0; mesh_id < scene->mNumMeshes; mesh_id++)
			{
				const aiMesh * ai_mesh = scene->mMeshes[mesh_id];
				if (ai_mesh->HasFaces()) {

					//std::cout << "num vertices :" << ai_mesh->mNumVertices << std::endl;

					for (size_t vert_id = 0; vert_id < ai_mesh->mNumVertices; vert_id++)
					{
						
						mesh.addVertex(ofVec3f(ai_mesh->mVertices[vert_id].x, ai_mesh->mVertices[vert_id].y, ai_mesh->mVertices[vert_id].z));
					}

					if (ai_mesh->HasNormals()) {
						ofLogNotice("ObjectImporter", "Mesh has Normals");
						//printf("Mesh has Normals\n");
						for (size_t n_id = 0; n_id < ai_mesh->mNumVertices; n_id++)
						{
							mesh.addNormal(ofVec3f(ai_mesh->mNormals[n_id].x, ai_mesh->mNormals[n_id].y, ai_mesh->mNormals[n_id].z));
							
						}
					}

					if (ai_mesh->HasTextureCoords(0)) {
						ofLogNotice("ObjectImporter", "Mesh has Textures Coordinates");
						//printf("Mesh has Textures Coordinates\n");
						for (size_t vert_id = 0; vert_id < ai_mesh->mNumVertices; vert_id++)
						{
							ofVec2f coord = ofVec2f(ai_mesh->mTextureCoords[0][vert_id].x, ai_mesh->mTextureCoords[0][vert_id].y);
							mesh.addTexCoord(coord);
							//ofLog(OF_LOG_NOTICE, ofToString(coord));
						}
					}
					else {
						ofLogNotice("ObjectImporter", "Mesh has NO Textures Coordinates");
						//printf("Mesh has NO Textures Coordinates\n");
					}

					for (size_t face_id	= 0; face_id < ai_mesh->mNumFaces; face_id++)
					{
						for (size_t i = 0; i < ai_mesh->mFaces[face_id].mNumIndices; i++)
						{
							mesh.addIndex(ai_mesh->mFaces[face_id].mIndices[i]);
						}
					}
				}
			}
		}

		
	}
	return mesh;
}
