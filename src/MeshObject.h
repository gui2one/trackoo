#pragma once

#include "ofMain.h"

class MeshObject : public ofNode {
public:
	
	MeshObject();
	


	virtual void customDraw() {
		//ofSphere(0, 0, 0, 100.0);
		if (m_mesh_ptr != nullptr) {
			
			
			ofEnableLighting();
			renderer->draw(*m_mesh_ptr, OF_MESH_FILL);
			ofDisableLighting();
			//m_mesh_ptr->draw();
			//m_mesh_ptr->drawWireframe();
			
			renderer->drawAxis(1.0);
			
		}

		
	}

	inline void setMesh(ofMesh * mesh_ptr) {
		m_mesh_ptr = mesh_ptr;
	}

	inline ofMesh * getMeshPtr() {
		return m_mesh_ptr;
	}

	shared_ptr<ofBaseGLRenderer> renderer;

private:

	ofMesh * m_mesh_ptr = nullptr;
};