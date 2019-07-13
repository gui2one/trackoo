#pragma once

#include "ofMain.h"

class MeshObject : public ofNode {
public:
	
	MeshObject();
	


	virtual void customDraw() {
		//ofSphere(0, 0, 0, 100.0);
		if (m_mesh_ptr != nullptr) {
			ofEnableLighting();
			m_mesh_ptr->draw();
			//m_mesh_ptr->drawWireframe();
			ofDisableLighting();
		}

		//ofDrawAxis(1.0);
	}

	inline void setMesh(ofMesh * mesh_ptr) {
		m_mesh_ptr = mesh_ptr;
	}

	inline ofMesh * getMeshPtr() {
		return m_mesh_ptr;
	}

private:

	ofMesh * m_mesh_ptr = nullptr;
};