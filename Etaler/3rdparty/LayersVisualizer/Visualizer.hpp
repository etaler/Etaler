#pragma once

#include <map>
#include <any>
#include <string>
#include "viewer_imgui.h"
#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/point_cloud.h>
#include <easy3d/util/timer.h>
#include <easy3d/viewer/drawable.h>

using namespace easy3d;

class  Visualizer
{
	std::string name;
	ViewerImGui * viewer;
	int initialized;
	PointCloud* gccloud;
	int width;
	int length;
	int height;
	//easy3d::Viewer * viewer;
 public:
 	Visualizer();
	Visualizer(int w, int l);
	void Update();
	int AddLayer(int width, int length, int height);
	void UpdateLayer(int idx, bool * buf);
    void StartViewer();
	void WaitForGLInitDone();
};
