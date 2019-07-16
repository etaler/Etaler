#include <fstream>
 #include <unistd.h>
#include <Visualizer.hpp>
#include "viewer_imgui.h"
#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/point_cloud.h>
#include <easy3d/util/timer.h>
#include <easy3d/viewer/drawable.h>
#include <thread> 
using namespace easy3d;


Visualizer::Visualizer()
{
	width = 100;
	length = 100;
	initialized = 0;
	std::thread th1(&Visualizer::StartViewer, this); 
	th1.detach();
	WaitForGLInitDone();
}

Visualizer::Visualizer(int w, int l)
{
	width = w;
	length = l;
	initialized = 0;
	std::thread th1(&Visualizer::StartViewer, this); 
	th1.detach();
	WaitForGLInitDone();
}
void Visualizer::StartViewer()
{
	viewer = new easy3d::ViewerImGui("Easy3D ImGui Viewer", 8, 3, 2); 
	viewer->resize(800, 600);
	//initialized = 1;
	gccloud = new easy3d::PointCloud;

	for (float i=0; i<length; i++){
		for (float j = 0; j <  width; j++) {
			gccloud->add_vertex(vec3(i, j, 0));
		}           
	}

	PointsDrawable* gc_drawable = gccloud->add_points_drawable("vertices");
	gc_drawable->set_per_vertex_color(true);
	// TODO: make the point size a function of the number of points 
	gc_drawable->set_point_size(10);
	auto gcpoints = gccloud->get_vertex_property<vec3>("v:point");
	auto gccolors = gccloud->add_vertex_property<vec3>("v:color");

	for (auto v : gccloud->vertices())	// iterate over all vertices
	{
		auto c = gcpoints[v];
		gccolors[v] =  vec3(255, 0, 0);         
	}
	gc_drawable->update_vertex_buffer(gcpoints.vector());
	gc_drawable->update_color_buffer(gccolors.vector());
	viewer->add_model(gccloud);
	initialized = 1;
	viewer->run();
}

void Visualizer::WaitForGLInitDone()
{
	while (initialized == 0)
	{
		usleep(1000);
	}
}
int Visualizer::AddLayer(int width, int length, int index)
{
	
    return 0;
}

void Visualizer::UpdateLayer(int idx, bool * buf)
{
	easy3d::Model* model = this->viewer->current_model();
	if (model)
	{
		easy3d::PointCloud* cloud = dynamic_cast<PointCloud*>(model);
		if (cloud)
		{
			// update vertices color
			easy3d::PointsDrawable* drawable = cloud->points_drawable("vertices");
			PointCloud::VertexProperty<vec3> colors = cloud->get_vertex_property<vec3>("v:color");
			auto gcpoints = gccloud->get_vertex_property<vec3>("v:point");
			for (auto v : cloud->vertices())	// iterate over all vertices
			{
				auto c = gcpoints[v];
				int flat_vec_location  = c[0]*width + c[1];
				if (buf[flat_vec_location])
				{
					colors[v] = vec3(1.0f , 0, 0);	
				}
				else
				{
					colors[v] = vec3(0 , 0, 0);
				}
			}		
		}
		else
		{
			std::cout << "Point cloud is null!" << std::endl;

		}
	}
	else
	{
		std::cout << "No GUI to update - did you close the GUI window ?" << std::endl;
	}
    return;
}


#if 0
//template <class Archive>
void add(Shape const & s)
{
	std::vector<int64_t> vec(s.begin(), s.end());
	//archive(vec);
}

template <class Archive>
void load(Archive & archive , Shape & s)
{
	std::vector<int64_t> vec;
	//archive(vec);
	//s = Shape(vec.begin(), vec.end());
}
#endif
#if 0
template <class Archive>
void save(Archive & archive, Tensor const & t)
{
	std::string dtype = [&t]() {
		if(t.dtype() == DType::Bool)
			return "uint8";
		if(t.dtype() == DType::Float)
			return "float";
		if(t.dtype() == DType::Int32)
			return "int32";

		throw EtError("Cannot handle such dtype()");
	}();


	archive(make_nvp("shape", t.shape()));
	archive(make_nvp("dtype", dtype));
	if(t.dtype() == DType::Bool) {
		std::vector<uint8_t> arr = t.toHost<uint8_t>();
		archive(make_nvp("data", arr));
	}
	else if(t.dtype() == DType::Float) {
		std::vector<float> arr = t.toHost<float>();
		archive(make_nvp("data", arr));
	}
	else if(t.dtype() == DType::Int32) {
		std::vector<int32_t> arr = t.toHost<int32_t>();
		archive(make_nvp("data", arr));
	}
}

template <class Archive>
void load(Archive & archive, Tensor & t)
{
	Shape s;
	archive(make_nvp("shape", s));

	std::string dtype;
	archive(make_nvp("dtype", dtype));

	if(dtype == "uint8") {
		std::vector<uint8_t> d(s.volume());
		archive(make_nvp("data", d));
		t = Tensor(s, d.data());
	}
	else if(dtype == "float") {
		std::vector<float> d(s.volume());
		archive(make_nvp("data", d));
		t = Tensor(s, d.data());
	}
	else if(dtype == "int32") {
		std::vector<int32_t> d(s.volume());
		archive(make_nvp("data", d));
		t = Tensor(s, d.data());
	}
}

template <class Archive>
void save(Archive & archive , std::vector<Tensor> const & v)
{
	archive(make_size_tag(static_cast<size_type>(v.size())));
	for(const auto& t : v)
		archive(t);
}

template <class Archive>
void load(Archive & archive , std::vector<Tensor> & v)
{
	size_type size;
	archive(make_size_tag(size));

	v.resize(size);
	for(auto& t : v)
		archive(t);
}

template <class Archive>
void save(Archive & archive ,StateDict const & item)
{
	std::vector<std::string> keys;
	std::vector<std::string> types;

	for(const auto & [k, v] : item)
		keys.push_back(k);
	archive(make_nvp("keys", keys));

	for(const auto & [k, v] : item) {
		if(v.type() == typeid(std::string))
			types.push_back("string");
		else if(v.type() == typeid(Shape))
			types.push_back("Shape");
		else if(v.type() == typeid(int32_t))
			types.push_back("int32_t");
		else if(v.type() == typeid(float))
			types.push_back("float");
		else if(v.type() == typeid(bool))
			types.push_back("bool");
		else if(v.type() == typeid(Tensor))
			types.push_back("Tensor");
		else if(v.type() == typeid(StateDict))
			types.push_back("StateDict");
		else if(v.type() == typeid(std::vector<Tensor>))
			types.push_back("std::vector<Tensor>");
		else if(v.type() == typeid(std::vector<int>))
			types.push_back("std::vector<int>");
		else if(v.type() == typeid(std::vector<float>))
			types.push_back("std::vector<float>");
		else
			throw EtError("Cannot save (mangled name:) type " + std::string(v.type().name()) + ", key " + k);
	}
	archive(make_nvp("types", types));

	for(const auto & [k, v] : item) {
		if(v.type() == typeid(std::string))
			archive(std::any_cast<std::string>(v));
		else if(v.type() == typeid(Shape))
			archive(std::any_cast<Shape>(v));
		else if(v.type() == typeid(int32_t))
			archive(std::any_cast<int32_t>(v));
		else if(v.type() == typeid(float))
			archive(std::any_cast<float>(v));
		else if(v.type() == typeid(bool))
			archive(std::any_cast<bool>(v));
		else if(v.type() == typeid(Tensor))
			archive(std::any_cast<Tensor>(v));
		else if(v.type() == typeid(StateDict))
			archive(std::any_cast<StateDict>(v));
		else if(v.type() == typeid(std::vector<Tensor>))
			archive(std::any_cast<std::vector<Tensor>>(v));
		else if(v.type() == typeid(std::vector<int>))
			archive(std::any_cast<std::vector<int>>(v));
		else if(v.type() == typeid(std::vector<float>))
			archive(std::any_cast<std::vector<float>>(v));
		else
			throw EtError("Cannot save type " + std::string(typeid(decltype(v)).name()) + ", key " + k);

	}
}

template <typename T, class Archive>
void read_archive(Archive & archive, StateDict& dict, std::string key)
{
	T v;
	archive(v);
	dict[key] = v;
}

template <class Archive>
void load(Archive & archive ,StateDict & item)
{
	std::vector<std::string> keys;
	std::vector<std::string> types;

	archive(make_nvp("keys", keys));
	archive(make_nvp("types", types));

	et_assert(keys.size() == types.size());

	for(size_t i=0;i<keys.size();i++) {
		std::string key = keys[i];
		std::string type = types[i];
		if(type == "string")
			read_archive<std::string>(archive, item, key);
		else if(type == "Shape")
			read_archive<Shape>(archive, item, key);
		else if(type == "int32_t")
			read_archive<int32_t>(archive, item, key);
		else if(type == "float")
			read_archive<float>(archive, item, key);
		else if(type == "bool")
			read_archive<bool>(archive, item, key);
		else if(type == "Tensor")
			read_archive<Tensor>(archive, item, key);
		else if(type == "StateDict")
			read_archive<StateDict>(archive, item, key);
		else if(type == "std::vector<Tensor>")
			read_archive<std::vector<Tensor>>(archive, item, key);
		else if(type == "std::vector<int>")
			read_archive<std::vector<int>>(archive, item, key);
		else if(type == "std::vector<float>")
			read_archive<std::vector<float>>(archive, item, key);
		else
			throw EtError("Cannot serealize type " + type);

	}
}

}

static std::string to_lower(std::string str)
{
	std::transform(str.begin(), str.end(), str.begin(), ::tolower);
	return str;
}

static std::string fileExtenstion(std::string path)
{
	size_t dot_pos = path.find_last_of('.');
	if(dot_pos == std::string::npos)
		return "";
	size_t slash_pos = path.find_last_of('/');
	if(slash_pos == std::string::npos)
		slash_pos = path.find_last_of('\\');
	if(slash_pos == std::string::npos || dot_pos > slash_pos)
		return to_lower(path.substr(dot_pos + 1));
	return "";
}

void et::save(const StateDict& dict, const std::string& path)
{
	std::ofstream out(path, std::ios::binary);

	std::string ext = fileExtenstion(path);
	if(ext == "json") {
		cereal::JSONOutputArchive ar(out);
		ar(dict);
	}
	else {
		cereal::PortableBinaryOutputArchive ar(out);
		ar(dict);
	}

}

StateDict et::load(const std::string& path)
{
	std::ifstream in(path, std::ios::binary);
	StateDict dict;

	std::string ext = fileExtenstion(path);
	if(ext == "json") {
		cereal::JSONInputArchive ar(in);
		ar(dict);
	}
	else {
		cereal::PortableBinaryInputArchive ar(in);
		ar(dict);
	}
	return dict;
}
#endif