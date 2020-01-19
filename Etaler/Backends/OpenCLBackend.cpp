#include "OpenCLBackend.hpp"

#include "Etaler/Core/Random.hpp"
#include "Etaler/Core/Views.hpp"
#include "Etaler/Core/String.hpp"

#include <map>
#include <sstream>
#include <fstream>

#include <stdlib.h>

using namespace et;

//Helper functions

static std::string readAll(const std::string& path)
{
	std::ifstream in(path);
	if(!in)
		throw EtError("Cannot open file " + path + ". " + strerror(errno));
	std::string str((std::istreambuf_iterator<char>(in)),
		std::istreambuf_iterator<char>());
	return str;
}

inline intmax_t selectWorkSize(intmax_t max, intmax_t mul_of, intmax_t size)
{
	auto round = [mul_of](auto v){return ((v/mul_of)*mul_of) + (v%mul_of == 0 ? 0 : mul_of);};
	return std::min((intmax_t)max, round(size));
}

#define OPENCL_TENSOR_MAX_DIMS 32
typedef struct __attribute__ ((packed)) _OpenCLView
{
        int stride[OPENCL_TENSOR_MAX_DIMS];
        int shape_stride[OPENCL_TENSOR_MAX_DIMS];
        int offset;
        int dims;
} OpenCLView;

static void makeOpenCLView(const TensorImpl* x, OpenCLView* v)
{
	int dims = int(x->dimentions());
	et_assert(dims <= OPENCL_TENSOR_MAX_DIMS, "Too much dimensions for OpenCL backend.");
	auto stride = x->stride();
	auto shape_stride = shapeToStride(x->shape());
	for(int i=0;i<dims;i++) {
		v->stride[i] = stride[i];
		v->shape_stride[i] = shape_stride[i];
	}
	v->offset = x->offset();
	v->dims = dims;
}



template <typename T>
std::string str(T&& s)
{
	return std::to_string(s);
}

OpenCLBackend::OpenCLBackend()
	: OpenCLBackend(0, 0)
{
}

OpenCLBackend::OpenCLBackend(size_t platform_id, size_t device_id)
{
	std::vector<cl::Platform> platforms;
	cl_int err = cl::Platform::get(&platforms);
	if(err != CL_SUCCESS)
		throw EtError("Failed to get OpenCL platforms. Error: " + std::to_string(err));
	if(platforms.size() == 0)
		throw EtError("No OpenCL platform found.");
	if(platforms.size() <= platform_id)
		throw EtError("OpenCL platform " + std::to_string(platform_id) + " not found.");
	auto& platform = platforms[platform_id];

	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if(devices.size() == 0)
		throw EtError("No OpenCL device found in platorm " + platform.getInfo<CL_PLATFORM_NAME>());
	if(devices.size() <= device_id)
		throw EtError("OpenCL device " + std::to_string(device_id) + " in platform " + platform.getInfo<CL_PLATFORM_NAME>() + " not found.");
	auto& device = devices[device_id];
	if(device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>() == CL_FALSE)
		throw EtError("Compiler for " + device.getInfo<CL_DEVICE_NAME>() + " is not avliable. (Devices like Altera/Xilinx FPGAs not supported"
			" in the OpenCL backend.)");

	cl::Context context = cl::Context(device, nullptr, nullptr, nullptr, &err);
	if(err != CL_SUCCESS)
		throw EtError("Failed to create OpenCL context. Error " + std::to_string(err));

	init(context, platform, device);
}

OpenCLBackend::OpenCLBackend(cl::Context context, cl::Platform platform, cl::Device device)
{
	init(context, platform, device);
}

void OpenCLBackend::init(cl::Context context, cl::Platform platform, cl::Device device)
{
	context_ = std::move(context);
	platform_ = std::move(platform);
	device_ = std::move(device);

	//I trust these won't fail
	queue_ = cl::CommandQueue(context_);
	kernel_manager_ = KernelManager(device_, context_);
	kernel_manager_.compileKernel("kernel void __etaler_dummy__(global int* p){p[get_global_id(0)] = 0;}", "__etaler_dummy__", "__etaler_dummy__");

	local_mem_size_ = device_.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
	local_mem_type_ = device_.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>();
	num_compute_units_ = device_.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

	cl_int err = 0;
	//Get the list of extention suuported
	std::string extentions = device_.getInfo<CL_DEVICE_EXTENSIONS>(&err);
	if(err != CL_SUCCESS)
		throw EtError("Failed to aquire supported OpenCL extention on device " +
			device_.getInfo<CL_DEVICE_NAME>() + ". Error " + std::to_string(err));
	supported_extentions_ = split(extentions, ' ');

	std::string device_name = device_.getInfo<CL_DEVICE_NAME>();

	//Make sure extentions used by Etaler is avaliable
	if(localMemoryType() == CL_LOCAL) {
		et_assert(isExtentionSupported("cl_khr_local_int32_base_atomics")
			, "The required exntention cl_khr_local_int32_base_atomics is not supported by " + device_name);
		et_assert(isExtentionSupported("cl_khr_local_int32_extended_atomics")
			, "The required exntention cl_khr_local_int32_extended_atomics is not supported by " + device_name);
	}

	have_fp16_ = isExtentionSupported("cl_khr_fp16");
}

std::shared_ptr<TensorImpl> OpenCLBackend::createTensor(const Shape& shape, DType dtype, const void* data)
{
	et_assert(dtype != DType::Unknown);
	size_t buf_size = shape.volume()*dtypeToSize(dtype);
	cl::Buffer buf = allocBuffer(buf_size);

	if(data != nullptr) {
		cl_int err;
		void* ocl_ptr = queue_.enqueueMapBuffer(buf, CL_TRUE, CL_MAP_WRITE, 0, buf_size, nullptr, nullptr, &err);
		if(err != CL_SUCCESS)
			throw EtError("OpenCL buffer map failed. Error: " + std::to_string(err) + ", map size " + std::to_string(buf_size));
		memcpy(ocl_ptr, data, buf_size);
		err = queue_.enqueueUnmapMemObject(buf, ocl_ptr, nullptr, nullptr);
		if(err != CL_SUCCESS)
			throw EtError("OpenCL buffer write failed. Error: " + std::to_string(err) + ", write size " + std::to_string(buf_size));
	}

	return createTensor(shape, dtype, buf);
}

std::shared_ptr<TensorImpl> OpenCLBackend::createTensor(const Shape& shape, DType dtype, cl::Buffer buf)
{
	if(dtype == DType::Half && have_fp16_ == false)
		throw EtError("Creating half(fp16) tensor but device have no fp16 capablity.");
	auto ptr = std::shared_ptr<OpenCLBuffer>(new OpenCLBuffer(shape, dtype, buf, shared_from_this()), [this](OpenCLBuffer* ptr){releaseTensor(ptr);});
	return std::make_shared<TensorImpl>(ptr, shape, shapeToStride(shape));
}

void OpenCLBackend::releaseTensor(OpenCLBuffer* buf)
{
	delete buf;
}

void OpenCLBackend::copyToHost(const TensorImpl* t, void* dest)
{
	auto b = std::static_pointer_cast<const OpenCLBuffer>(t->buffer());
	if(b == nullptr)
		throw EtError("Cannot copy to host memory: Tensor/Backend mismach");
	cl_int err = queue_.enqueueReadBuffer(b->buffer(), CL_TRUE, 0, t->size()*dtypeToSize(t->dtype()), dest);
	if(err != CL_SUCCESS)
		throw EtError("OpenCL buffer readback failed. Error: " + std::to_string(err));
}

std::string OpenCLBackend::deviceInfo() const
{
	std::map<int, std::string> local_type;
	local_type[CL_LOCAL] = "Local";
	local_type[CL_GLOBAL] = "Global";
	local_type[CL_NONE] = "None";
	std::string res;
	res += "Platform: " + platform_.getInfo<CL_PLATFORM_NAME>() + "\n";
	res += "Device name: " + device_.getInfo<CL_DEVICE_NAME>() + "\n";
	res += "Global memory size: " + std::to_string((float)device_.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/1024/1024) + " GB\n";
	res += "Max allocatable memory: " + std::to_string((float)device_.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()/1024/1024) + " GB\n";
	res += "Local memory size: " + std::to_string(localMemorySize()/1024) + " KB\n";
	res += "Local memory type: " + local_type[localMemoryType()] + "\n";
	res += "Prefered work group size: " + std::to_string(kernel_manager_.kernel("__etaler_dummy__").getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device_)) + "\n";
	res += "Half percision: " + std::string(isExtentionSupported("cl_khr_fp16") ? "Yes" : "No");
	return res;
}

KernelManager::KernelManager(cl::Device device, cl::Context context)
	: device_(device), context_(context)
{
	const char* ptr = getenv("ETALER_KERNEL_PATH");
	if(ptr == nullptr)
		return;
	std::string env_path(ptr);
	if(env_path != "") {
		if(env_path.back() != '/')
			env_path += "/";
		addSearchPath(env_path);
	}
}

cl::Kernel KernelManager::compileKernel(const std::string& src, const std::string& program_name, const std::string& kernel_name
	, bool force_override, const std::string& flags)
{
	compileKernel(src, program_name, std::vector<std::string>{kernel_name}, force_override, flags);
	assert(exists(program_name, kernel_name)!=false);
	return kernel(program_name, kernel_name);
}

void KernelManager::compileKernel(const std::string& src, const std::string& program_name, const std::vector<std::string>& kernel_names
	, bool force_override, const std::string& flags)
{
	compileKernel(std::vector<std::string>{src}, program_name, kernel_names, force_override, flags);
}

void KernelManager::compileKernel(const std::vector<std::string>& srcs, const std::string& program_name, const std::vector<std::string>& kernel_names
	, bool force_override, const std::string& flags)
{
	if(apps_.find(program_name) != apps_.end() && force_override == false)
		return;
	cl::Program::Sources sources;
	for(const auto& src : srcs)
		sources.push_back({src.c_str(), src.size()});

	cl::Program program(context_,sources);
	cl_int err = program.build({device_}, flags.c_str());
	if(err != CL_SUCCESS) {
		throw EtError("Error building OpenCL program: " + program_name
			+ ", Error:" + std::to_string(err) +"\n" + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_));
	}

	auto& app = apps_[program_name];
	for(const auto& name : kernel_names) {
		cl_int err;
		cl::Kernel k(program, name.c_str(), &err);
		if(err != CL_SUCCESS)
			throw EtError("Kernel " + name + " not found in program " + program_name);
		app.kernels[name] = k;
	}
}

void KernelManager::compileFromFile(const std::string& path, const std::string& program_name, const std::vector<std::string>& kernel_names
	, bool force_override, const std::string& flags, const std::string& prepend)
{
	compileFromFile(std::vector<std::string>{path}, program_name, kernel_names, force_override, flags, prepend);
}

void KernelManager::compileFromFile(const std::vector<std::string>& paths, const std::string& program_name, const std::vector<std::string>& kernel_names
	, bool force_override, const std::string& flags, const std::string& prepend)
{
	std::vector<std::string> sources;
	for(const auto& path : paths)
		sources.emplace_back(prepend + (prepend!=""?"\n":"") + readKernel(path));
	compileKernel(sources, program_name, kernel_names, force_override, flags);
}

std::string KernelManager::readKernel(const std::string& name)
{
	auto it = kernelCache.find(name);
	if(it != kernelCache.end())
		return it->second;

	for(const auto& search_paths : search_paths_) {
		std::string path = search_paths + name;
		if(std::ifstream(path).is_open() == true) {
			std::string src = readAll(path);
			kernelCache[name] = src;
			return src;
		}
	}

	throw EtError("Cannot find any open-able " + name + " in search paths");
}

void KernelManager::addSearchPath(const std::string& path)
{
	search_paths_.push_front(path);
}

std::shared_ptr<TensorImpl> OpenCLBackend::cellActivity(const TensorImpl* x, const TensorImpl* connections,
	const TensorImpl* permeances, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse)
{
	requireProperties(x, this, DType::Bool, IsPlain());
	requireProperties(connections, this, DType::Int32, IsPlain());
	requireProperties(permeances, this, IsDType{DType::Float, DType::Half}, IsPlain());
	et_assert(connections->shape() == permeances->shape());
	et_assert(connections->dimentions() >= 2);

	Shape s = connections->shape();
	s.pop_back();
	auto y = createTensor(s, DType::Int32);

	auto param_hash = hashify(x->size(), connections->shape().back(), !has_unconnected_synapse, permeances->dtype());
	auto program_name = "cellActivity"+param_hash;
	if(kernel_manager_.exists(program_name) == false) {

		auto args = "-DINPUT_SIZE="+str(x->size())+" -DMAX_SYNAPSE_PER_CELL="+str(connections->shape().back())+" -DNO_UNUSED_SYNAPSE=" +
			str(!has_unconnected_synapse) + " -DPERM_TYPE="+to_ctype_string(permeances->dtype());
		auto prepend = (permeances->dtype()==DType::Half?"#pragma OPENCL EXTENSION cl_khr_fp16 : enable":"");

		std::string kernel_file = "";
		if(x->size() < localMemorySize() && localMemoryType() == CL_LOCAL)
			kernel_file = "cellActivity.cl";
		else if(x->size() < localMemorySize()*8-8 && localMemoryType() == CL_LOCAL)
			kernel_file = "cellActivity_compressed_local.cl";
		else
			kernel_file = "cellActivity_global.cl";
		assert(kernel_file != "");
		kernel_manager_.compileFromFile(kernel_file, program_name, {"cellActivity"}, false, args, prepend);
	}
	cl::Kernel k = kernel_manager_.kernel(program_name, "cellActivity");

	k.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<const OpenCLBuffer>(connections->buffer())->buffer());
	k.setArg(2, std::static_pointer_cast<const OpenCLBuffer>(permeances->buffer())->buffer());
	k.setArg(3, std::static_pointer_cast<OpenCLBuffer>(y->buffer())->buffer());
	k.setArg(4, (float)connected_permeance);
	k.setArg(5, (int)active_threshold);
	k.setArg(6, (int)y->size());

	size_t local_size = 64;
	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, x->size())), cl::NDRange(local_size));

	if(err != CL_SUCCESS)
		throw EtError("overlapScore kernel overlapScore execution failed. Code " + str(err));

	return y;
}

std::shared_ptr<TensorImpl> OpenCLBackend::globalInhibition(const TensorImpl* x, float fraction)
{
	requireProperties(x, this, DType::Int32, IsPlain());

	auto y = createTensor(x->shape(), DType::Bool);

	auto param_hash = hashify(x->size(), 2000);
	auto program_name = "globalInhibition"+param_hash;
	if(kernel_manager_.exists(program_name) == false) {
		auto args = "-DINPUT_SIZE="+str(x->size())+" -DMAX_INPUT_VALUE="+str(2000);
		kernel_manager_.compileFromFile("globalInhibition.cl", program_name, {"fastTopK", "threshold"}, false, args);
	}

	cl::Kernel topKKernel, thresholdKernel;
	topKKernel = kernel_manager_.kernel(program_name, "fastTopK");
	thresholdKernel = kernel_manager_.kernel(program_name, "threshold");

	cl::Buffer threshold = allocBuffer(sizeof(int));

	topKKernel.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer());
	topKKernel.setArg(1, threshold);
	topKKernel.setArg(2, (int)(x->size()*fraction));

	queue_.enqueueNDRangeKernel(topKKernel, cl::NullRange, cl::NDRange(256), cl::NDRange(256));

	thresholdKernel.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer());
	thresholdKernel.setArg(1, std::static_pointer_cast<const OpenCLBuffer>(y->buffer())->buffer());
	thresholdKernel.setArg(2, threshold);
	queue_.enqueueNDRangeKernel(thresholdKernel, cl::NullRange, cl::NDRange(1024), cl::NDRange(32));

	return y;
}

std::shared_ptr<TensorImpl> OpenCLBackend::cast(const TensorImpl* x, DType toType)
{
	requireProperties(x, this, IsPlain());
	auto param_hash = hashify(x->dtype(), toType, x->dtype() == DType::Half || toType == DType::Half);
	auto program_name = "cast"+param_hash;
	if(kernel_manager_.exists(program_name) == false) {
		auto args = "-DInType="+to_ctype_string(x->dtype())+" -DOutType="+to_ctype_string(toType)
			+ (x->dtype() == DType::Half || toType == DType::Half ? " -DHalfSupport" : "");
		kernel_manager_.compileFromFile("cast.cl", program_name, {"cast"}, false, args);
	}

	cl::Kernel k = kernel_manager_.kernel(program_name, "cast");

	auto res = createTensor(x->shape(), toType);
	k.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<OpenCLBuffer>(res->buffer())->buffer());
	k.setArg(2, (int)x->size());
	queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1024), cl::NDRange(32));

	return res;
}

void OpenCLBackend::sync() const
{
	cl_int err = queue_.finish();
	if(err != CL_SUCCESS)
		throw EtError("Waiting for OpenCL queue failed. Error: " + std::to_string(err));
}

std::shared_ptr<TensorImpl> OpenCLBackend::copy(const TensorImpl* x)
{
	requireProperties(x, this, IsContingous());
	size_t buf_size = x->size()*dtypeToSize(x->dtype());
	size_t offset = x->offset()*dtypeToSize(x->dtype());
	cl::Buffer buf = allocBuffer(buf_size);
	const cl::Buffer& src = std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer();
	cl_int err = queue_.enqueueCopyBuffer(src, buf, offset, 0, buf_size);
	if(err != CL_SUCCESS)
		throw EtError("Data copy enqueuing failed. Error " + std::to_string(err));

	return createTensor(x->shape(), x->dtype(), buf);
}

void OpenCLBackend::learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections,
	TensorImpl* permeances, float perm_inc, float perm_dec, bool has_unconnected_synapse)
{
	requireProperties(x, this, DType::Bool, IsPlain());
	requireProperties(learn, this, DType::Bool, IsPlain());
	requireProperties(connections, this, DType::Int32, IsPlain());
	requireProperties(permeances, this, IsDType{DType::Float, DType::Half}, IsPlain());

	et_assert(connections->shape() == permeances->shape());

	auto param_hash = hashify(x->size(), connections->shape().back(), !has_unconnected_synapse, learn->size(), permeances->dtype());
	auto program_name = "learnCorrilation"+param_hash;

	if(kernel_manager_.exists(program_name) == false) {
		auto args = "-DINPUT_SIZE="+str(x->size())+" -DMAX_SYNAPSE_PER_CELL="+str(connections->shape().back()) +
			" -DNO_UNUSED_SYNAPSE="+str(!has_unconnected_synapse)+" -DOUTPUT_SIZE="+str(learn->size()) +
			" -DPERM_TYPE="+to_ctype_string(permeances->dtype());
		auto prepend = (permeances->dtype()==DType::Half?"#pragma OPENCL EXTENSION cl_khr_fp16 : enable":"");

		std::string kernel_file = "";
		if(x->size() < localMemorySize() && localMemoryType() == CL_LOCAL)
			kernel_file = "learnCorrilation.cl";
		else if(x->size() < localMemorySize()*8-8 && localMemoryType() == CL_LOCAL)
			kernel_file = "learnCorrilation_compressed_local.cl";
		else
			kernel_file = "learnCorrilation_global.cl";
		assert(kernel_file != "");
		kernel_manager_.compileFromFile(kernel_file, program_name, {"learnCorrilation"}, false, args, prepend);
		
	}
	cl::Kernel k = kernel_manager_.kernel(program_name, "learnCorrilation");

	k.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<const OpenCLBuffer>(learn->buffer())->buffer());
	k.setArg(2, std::static_pointer_cast<const OpenCLBuffer>(connections->buffer())->buffer());
	k.setArg(3, std::static_pointer_cast<OpenCLBuffer>(permeances->buffer())->buffer());
	k.setArg(4, (float)perm_inc);
	k.setArg(5, (float)perm_dec);

	size_t local_size = 128;

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, x->size())), cl::NDRange(local_size));

	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel learnCorrilation execution failed. Code " + str(err));
}

void OpenCLBackend::sortSynapse(TensorImpl* connections, TensorImpl* permeances)
{
	requireProperties(connections, this, DType::Int32, IsPlain());
	requireProperties(permeances, this, IsDType{DType::Float, DType::Int32}, IsPlain());
	et_assert(connections->shape() == permeances->shape());

	auto param_hash = hashify(connections->shape().back(), permeances->dtype(), permeances->dtype()==DType::Half);
	auto program_name = "sortSynapse"+param_hash;
	if(kernel_manager_.exists(param_hash) == false) {
		auto args = "-DMAX_SYNAPSE_PER_CELL="+str(connections->shape().back()) + " -DPERM_TYPE="+to_ctype_string(permeances->dtype());
		auto prepend = (permeances->dtype()==DType::Half?"#pragma OPENCL EXTENSION cl_khr_fp16 : enable":"");
		kernel_manager_.compileFromFile("sort.cl", program_name, {"sortSynapse"}, false, args, prepend);
	}

	cl::Kernel k = kernel_manager_.kernel(program_name, "sortSynapse");

	int num_cells = connections->size()/connections->shape().back();

	cl::Buffer aux_buffer1 = allocBuffer(connections->size()*sizeof(int));
	cl::Buffer aux_buffer2 = allocBuffer(permeances->size()*sizeof(float));

	k.setArg(0, std::static_pointer_cast<OpenCLBuffer>(connections->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<OpenCLBuffer>(permeances->buffer())->buffer());
	k.setArg(2, num_cells);
	k.setArg(3, aux_buffer1);
	k.setArg(4, aux_buffer2);
	size_t local_size = 128;

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, num_cells)), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel execution failed. Code " + str(err));
}

std::shared_ptr<TensorImpl> OpenCLBackend::burst(const TensorImpl* x, const TensorImpl* s)
{
	requireProperties(x, this, DType::Bool, IsPlain());
	requireProperties(s, this, DType::Bool, IsPlain());

	Shape shape = s->shape();
	shape.pop_back();
	et_assert(shape == x->shape());

	auto res = copy(s);

	size_t num_columns = shape.volume();

	auto param_hash = hashify(s->shape().back(), num_columns);
	auto program_name = "applyBurst"+param_hash;
	if(kernel_manager_.exists(param_hash) == false) {
		auto args = "-DCELLS_PER_COLUMN="+str(s->shape().back())+" -DNUM_COLUMNS="+str(num_columns);
		kernel_manager_.compileFromFile("applyBurst.cl", program_name, {"applyBurst"}, false, args);
	}

	cl::Kernel k = kernel_manager_.kernel(program_name, "applyBurst");

	k.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<OpenCLBuffer>(res->buffer())->buffer());

	size_t local_size = 128;
	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, num_columns)), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel applyBurst execution failed. Code " + str(err));
	return res;
}

std::shared_ptr<TensorImpl> OpenCLBackend::reverseBurst(const TensorImpl* x)
{
	requireProperties(x, this, DType::Bool, IsPlain());

	size_t cells_per_column = x->shape().back();
	size_t num_columns = x->size()/cells_per_column;
	static pcg32 rng(42); //Static so the behavor hangees every time, breaking symmetry
	std::uniform_int_distribution<size_t> dist(0, cells_per_column-1);

	auto res = copy(x);

	intmax_t local_size = 128;
	intmax_t global_size = selectWorkSize(4096, local_size, num_columns);
	std::vector<uint32_t> seed1(global_size);
	std::vector<uint32_t> seed2(global_size);

	for(auto& v : seed1) v = rng();

	auto s1 = createTensor({global_size}, DType::Int32, seed1.data());
	auto s2 = createTensor({global_size}, DType::Int32, seed2.data());

	auto param_hash = hashify(cells_per_column, num_columns);
	auto program_name = "reverseBurst"+param_hash;
	if(kernel_manager_.exists(param_hash) == false) {
		auto args = "-DCELLS_PER_COLUMN="+str(cells_per_column)+" -DNUM_COLUMNS="+str(num_columns);
		kernel_manager_.compileFromFile("reverseBurst.cl", program_name, {"reverseBurst"}, false, args);
	}
	cl::Kernel k = kernel_manager_.kernel(program_name, "reverseBurst");

	k.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(res->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<const OpenCLBuffer>(s1->buffer())->buffer());
	k.setArg(2, std::static_pointer_cast<const OpenCLBuffer>(s2->buffer())->buffer());

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel reverseBurst execution failed. Code " + str(err));
	return res;
}

void OpenCLBackend::growSynapses(const TensorImpl* x, const TensorImpl* y, TensorImpl* connections
		, TensorImpl* permeances, float initial_perm)
{
	requireProperties(x, this, DType::Bool, IsPlain());
	requireProperties(y, this, DType::Bool, IsPlain());
	requireProperties(connections, this, DType::Int32, IsPlain());
	requireProperties(permeances, this, IsDType{DType::Float, DType::Int32}, IsPlain());

	et_assert(connections->shape() == permeances->shape());
	Shape s = connections->shape();
	s.pop_back();
	et_assert(s == y->shape());

	size_t max_synapses_per_cell = connections->shape().back();
	size_t input_cell_count = x->size();

	auto param_hash = hashify(y->size(), x->size(), max_synapses_per_cell, permeances->dtype());
	auto program_name = "growSynapses"+param_hash;

	if(kernel_manager_.exists(program_name) == false) {
		auto args = "-DNUM_CELLS="+str(y->size())+" -DNUM_INPUT_BITS="+str(x->size())+" -DMAX_SYNAPSE_PER_CELL="+str(max_synapses_per_cell)
			+" -DPERM_TYPE="+to_ctype_string(permeances->dtype());
		auto prepend = (permeances->dtype()==DType::Half?"#pragma OPENCL EXTENSION cl_khr_fp16 : enable":"");
		
		kernel_manager_.compileFromFile("growSynapses.cl", program_name, {"growSynapses"}, false, args, prepend);
	}

	cl::Kernel k = kernel_manager_.kernel(program_name, "growSynapses");

	size_t local_size = 32;
	size_t work_size = selectWorkSize(4096, local_size, y->size());
	size_t num_groups = work_size/local_size;

	static cl::Buffer aux = allocBuffer(input_cell_count*num_groups);
	if(input_cell_count*num_groups > aux.getInfo<CL_MEM_SIZE>())
		aux = allocBuffer(input_cell_count*num_groups);
	auto sparse = toSparse(x);
	if(sparse.has_value() == false)
		return;
	cl::Buffer sparse_x = sparse.value();

	int sparse_size = sparse_x.getInfo<CL_MEM_SIZE>()/sizeof(int);
	if(sparse_size == 0)
		return;

	k.setArg(0, sparse_x);
	k.setArg(1, std::static_pointer_cast<const OpenCLBuffer>(y->buffer())->buffer());
	k.setArg(2, std::static_pointer_cast<OpenCLBuffer>(connections->buffer())->buffer());
	k.setArg(3, std::static_pointer_cast<OpenCLBuffer>(permeances->buffer())->buffer());
	k.setArg(4, initial_perm);
	k.setArg(5, sparse_size);
	k.setArg(6, aux);

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(work_size), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel growSynapses execution failed. Code " + str(err));
}

std::optional<cl::Buffer> OpenCLBackend::toSparse(const TensorImpl* x)
{
	requireProperties(x, this, DType::Bool, IsPlain());

	auto param_hash = hashify(x->size());
	auto program_name = "toSparse"+param_hash;
	if(kernel_manager_.exists(program_name) == false) {
		auto args = "-DINPUT_SIZE="+str(x->size());
		kernel_manager_.compileFromFile("toSparse.cl", program_name, {"toSparse", "onBits"}, false, args);
	}

	cl::Kernel on_bits = kernel_manager_.kernel(program_name, "onBits");
	cl::Kernel k = kernel_manager_.kernel(program_name, "toSparse");

	cl::Buffer num = allocBuffer(sizeof(int));
	cl::Buffer x_buf = std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer();

	on_bits.setArg(0, x_buf);
	on_bits.setArg(1, num);
	cl_int err = queue_.enqueueNDRangeKernel(on_bits, cl::NullRange, cl::NDRange(256), cl::NDRange(256));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel onBits execution failed. Code " + str(err));

	int* num_on = (int*) queue_.enqueueMapBuffer(num, CL_TRUE, CL_MAP_READ, 0, sizeof(int));
	int num_elements = *num_on;
	queue_.enqueueUnmapMemObject(num, num_on);

	if(num_elements == 0)
		return {};

	cl::Buffer buf = allocBuffer(num_elements*sizeof(int));
	k.setArg(0, x_buf);
	k.setArg(1, buf);
	err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(256), cl::NDRange(256));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel toSparse execution failed. Code " + str(err));
	return buf;
}

static std::string jitStridedView(const TensorImpl* x, size_t id)
{
	// If possible, do the easy route
	if(x->iscontiguous()) {
		std::string func = R"(
		int location_func$ID(int index) {
			return index + $OFFSET;	
		}
		)";
		replaceAll(func, "$ID", std::to_string(id));
		replaceAll(func, "$OFFSET", std::to_string(x->offset()));
		return func;
	}

	// Otherwise go the complex one
	std::string func = R"(
int location_func$ID(int index)
{
	int shape_stride[] = $SHAPE_STRIDE;
	int stride[] = $STRIDE;
	int offset = $OFFSET;
	int curr_idx = index;
	int sum = 0;
	for(int i=0;i<$DIMS;i++) {
		int s = shape_stride[i];
		int ndpos = curr_idx / s;
		sum += ndpos * stride[i];
		curr_idx %= s;
	}
	return sum + offset;
}
)";
	const auto shape_stride = shapeToStride(x->shape());
	replaceAll(func, "$ID", std::to_string(id));
	replaceAll(func, "$SHAPE_STRIDE", to_string(shape_stride));
	replaceAll(func, "$DIMS", std::to_string(x->dimentions()));
	replaceAll(func, "$STRIDE", to_string(x->stride()));
	replaceAll(func, "$OFFSET", std::to_string(x->offset()));
	return func;
}

std::shared_ptr<TensorImpl> OpenCLBackend::realize(const TensorImpl* x)
{
	requireProperties(x, this);
	if(x->isplain() == true)
		return copy(x);
	
	auto res = createTensor(x->shape(), x->dtype());
	assign(res.get(), x);
	return res;
}


void OpenCLBackend::assign(TensorImpl* dest, const TensorImpl* src)
{
	requireProperties(dest, this);
	requireProperties(src, this);

	if(dest->shape() != src->shape()) {
		throw EtError("Shape mismatch in tensor assignment. Shape "
			+ to_string(dest->shape()) + " and " + to_string(src->shape()));
	}

	auto param_hash = hashify(dest->dtype(), src->dtype());
	auto program_name = "copy"+param_hash;
	if(kernel_manager_.exists(program_name) == false) {
		auto args = "-DINPUT_TYPE="+to_ctype_string(src->dtype())+" -DOUTPUT_TYPE="+to_ctype_string(dest->dtype());
		kernel_manager_.compileFromFile("copy.cl", program_name, {"copy"}, false, args);
	}
	cl::Kernel k = kernel_manager_.kernel(program_name, "copy");

	OpenCLView input_view;
	OpenCLView output_view;
	makeOpenCLView(src, &input_view);
	makeOpenCLView(dest, &output_view);

	k.setArg(0, std::static_pointer_cast<OpenCLBuffer>(dest->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<const OpenCLBuffer>(src->buffer())->buffer());
	k.setArg(2, output_view);
	k.setArg(3, input_view);
	k.setArg(4, int(src->size()));


	size_t local_size = 128;
	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, src->size())), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel execution failed. Code " + str(err));
}

std::shared_ptr<TensorImpl> OpenCLBackend::sum(const TensorImpl* x, size_t chunk_size, DType dtype)
{
	requireProperties(x, this, IsPlain());
	et_assert(x->size() % chunk_size == 0);

	DType result_dtype = dtype;
	if(dtype == DType::Unknown) {
		result_dtype = [x](){
			DType dtype = x->dtype();
			if(dtype == DType::Bool || dtype == DType::Int32)
				return DType::Int32;
			else if(dtype == DType::Half)
				return DType::Half;
			else
				return DType::Float;
		}();
	}

	DType intermid_type = [](DType in, DType out) {
		if(in == DType::Float)
			return DType::Float;
		if(out == DType::Half)
			return DType::Half;
		return DType::Int32;
	}(x->dtype(), result_dtype);

	intmax_t result_size = intmax_t(x->size()/chunk_size);
	bool use_local_kernel = result_size <= numComputeUnits(); // Weather to use the kernel optimized for generating small number of results
	auto param_hash = hashify(x->dtype(), result_dtype, intermid_type, use_local_kernel);
	std::string program_name = "sum" + param_hash;
	if(kernel_manager_.exists(program_name) == false) {
		std::string args = "-DInType=" + to_ctype_string(x->dtype()) + " -DOutType=" + to_ctype_string(result_dtype) + " -DIntermidType=" + to_ctype_string(intermid_type)
		+ (intermid_type==DType::Half? " -DIntermidIsHalf" : "");

		if(use_local_kernel)
			kernel_manager_.compileFromFile("sum_local.cl", program_name, {"sum"}, false, args);
		else
			kernel_manager_.compileFromFile("sum.cl", program_name, {"sum"}, false, args);
	}

	cl::Kernel k = kernel_manager_.kernel(program_name, "sum");
	auto res = createTensor({result_size}, result_dtype);

	k.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<OpenCLBuffer>(res->buffer())->buffer());
	k.setArg(2, int(x->size()));
	k.setArg(3, int(chunk_size));

	cl_int err = CL_SUCCESS;
	if(use_local_kernel) {
		size_t local_size = 64; // the same value set in sum_local.cl
		err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(local_size*result_size), cl::NDRange(local_size));
	}
	else {
		size_t local_size = 128;
		err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, x->size()/chunk_size)), cl::NDRange(local_size));
	}
	
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel execution failed. Code " + str(err));
	return res;
}

void OpenCLBackend::decaySynapses(TensorImpl* connections, TensorImpl* permeances, float threshold)
{
	requireProperties(connections, this, DType::Int32, IsPlain());
	requireProperties(permeances, this, IsDType{DType::Float, DType::Half}, IsPlain());
	et_assert(connections->shape() == permeances->shape());

	size_t max_synapses_per_cell = connections->shape().back();
	size_t input_cell_count = connections->size()/max_synapses_per_cell;

	auto param_hash = hashify(input_cell_count, max_synapses_per_cell, permeances->dtype());
	std::string program_name = "decaySynapses" + param_hash;
	if(kernel_manager_.exists(program_name) == false) {
		auto args = "-DNUM_CELLS="+str(input_cell_count) + " -DMAX_SYNAPSE_PER_CELL="+str(max_synapses_per_cell) + 
			" -DPERM_TYPE="+to_ctype_string(permeances->dtype());
		auto prepend = (permeances->dtype()==DType::Half?"#pragma OPENCL EXTENSION cl_khr_fp16 : enable":"");
		kernel_manager_.compileFromFile("decaySynapses.cl", program_name, {"decaySynapses"}, false, args, prepend);
	}

	cl::Kernel k = kernel_manager_.kernel(program_name, "decaySynapses");

	k.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(connections->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<const OpenCLBuffer>(permeances->buffer())->buffer());
	k.setArg(2, threshold);

	size_t local_size = 128;

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, input_cell_count)), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel execution failed. Code " + str(err));
}

static std::string jitUniaryOperation(const TensorImpl* x, std::string f)
{
	std::string kernel = R"(

kernel void op(global T0* restrict x, global ResType* restrict y)
{
	int global_id = get_global_id(0);
	int global_size = get_global_size(0);
	for(int i=global_id;i<$SIZE;i+=global_size) {
		int position = location_func0(i);
		y[i] = f(x[position]);
	}
}

)";

	std::string extention_decl;
	if(x->dtype() == DType::Half)
		extention_decl = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable";

	std::string res = extention_decl + "\n" + f + "\n" + jitStridedView(x, 0) + "\n" + kernel;
	replaceAll(res, "$SIZE", std::to_string(x->size()));
	return res;
}

static std::string jitBinaryOperation(const TensorImpl* x1,const TensorImpl* x2 , std::string f)
{
	std::string kernel = R"(

kernel void op(global T0* restrict x1, global T1* restrict x2, global ResType* restrict y)
{
	int global_id = get_global_id(0);
	int global_size = get_global_size(0);
	for(int i=global_id;i<$SIZE;i+=global_size) {
		int p1 = location_func0(i);
		int p2 = location_func1(i);
		y[i] = f(x1[p1], x2[p2]);
	}
}
)";

	std::string extention_decl;
	if(x1->dtype() == DType::Half || x2->dtype() == DType::Half)
		extention_decl = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable";

	std::string res = extention_decl + "\n" + f + "\n" + jitStridedView(x1, 0) + "\n"  + jitStridedView(x2, 1) + "\n" + kernel;
	replaceAll(res, "$SIZE", std::to_string(x1->size()));
	return res;
}

std::shared_ptr<TensorImpl> OpenCLBackend::applyUnaryOp(const TensorImpl* x, std::string f, DType resType)
{
	requireProperties(x, this);

	auto param_hash = hashify(x->dtype(), resType, x->offset(), x->shape(), x->stride());
	std::string program_name = f+param_hash;
	if(kernel_manager_.exists(program_name) == false) {
		std::string args = "-DT0="+to_ctype_string(x->dtype())+" -DResType="+to_ctype_string(resType);
		std::string source = jitUniaryOperation(x, f);
		kernel_manager_.compileKernel(source, program_name, "op", false, args);
	}

	cl::Kernel k = kernel_manager_.kernel(program_name, "op");

	auto res = createTensor(x->shape(), resType);
	k.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(x->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<OpenCLBuffer>(res->buffer())->buffer());

	size_t local_size = 128;

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, x->size())), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel execution failed. Code " + str(err));

	return res;
}

std::shared_ptr<TensorImpl> OpenCLBackend::applyBinaryOp(const TensorImpl* x1, const TensorImpl* x2, std::string f, DType resType)
{
	requireProperties(x1, this);
	requireProperties(x2, this);
	et_assert(x1->shape() == x2->shape());

	auto to_str = [](auto x){
		return std::to_string(x->offset())+to_string(x->shape())+to_string(x->stride())+to_string(x->dtype());
	};

	auto param_hash = hashify(resType, to_str(x1), to_str(x2));
	std::string program_name = f+param_hash;
	if(kernel_manager_.exists(program_name) == false) {
		std::string args = "-DT0="+to_ctype_string(x1->dtype())+" -DT1="+to_ctype_string(x2->dtype()) + " -DResType="+to_ctype_string(resType);
		std::string source = jitBinaryOperation(x1, x2, f);
		kernel_manager_.compileKernel(source, program_name, "op", false, args);
	}

	cl::Kernel k = kernel_manager_.kernel(program_name, "op");

	auto res = createTensor(x1->shape(), resType);
	k.setArg(0, std::static_pointer_cast<const OpenCLBuffer>(x1->buffer())->buffer());
	k.setArg(1, std::static_pointer_cast<const OpenCLBuffer>(x2->buffer())->buffer());
	k.setArg(2, std::static_pointer_cast<OpenCLBuffer>(res->buffer())->buffer());

	size_t local_size = 128;

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, x1->size())), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel execution failed. Code " + str(err));

	return res;
}

std::shared_ptr<TensorImpl> OpenCLBackend::abs(const TensorImpl* x)
{
	DType result_type = [&x](){
		auto dtype = x->dtype();
		if(dtype == DType::Bool)
			return DType::Int32;
		return dtype;
	}();
	const char* func = (result_type == DType::Float || result_type == DType::Half ? "#define f(x) (fabs((ResType)x))" : "#define f(x) (abs((ResType)x))");
	return applyUnaryOp(x, func, result_type);
}

std::shared_ptr<TensorImpl> OpenCLBackend::exp(const TensorImpl* x)
{
	DType result_type = x->dtype() == DType::Half ? DType::Half : DType::Float;
	return applyUnaryOp(x, "#define f(x) (exp((ResType)x))", result_type);
}

std::shared_ptr<TensorImpl> OpenCLBackend::negate(const TensorImpl* x)
{
	DType result_type = x->dtype() == DType::Bool ? DType::Int32 : x->dtype();
	return applyUnaryOp(x, "#define f(x) (-x)", result_type);
}

std::shared_ptr<TensorImpl> OpenCLBackend::inverse(const TensorImpl* x)
{
	DType result_type = x->dtype() == DType::Half ? DType::Half : DType::Float;
	return applyUnaryOp(x, "#define f(x) ((ResType)1/(ResType)x)", result_type);
}

std::shared_ptr<TensorImpl> OpenCLBackend::log(const TensorImpl* x)
{
	DType result_type = x->dtype() == DType::Half ? DType::Half : DType::Float;
	return applyUnaryOp(x, "#define f(x) (log((ResType)x))", result_type);
}

std::shared_ptr<TensorImpl> OpenCLBackend::logical_not(const TensorImpl* x)
{
	return applyUnaryOp(x, "#define f(x) (!((bool)x))", DType::Bool);
}

static DType solveBinaryOpDType(DType t1, DType t2)
{
	if(t1 == DType::Float || t2 == DType::Float)
		return DType::Float;
	else if(t1 == DType::Half || t2 == DType::Half)
		return DType::Half;
	return DType::Int32;
}

std::shared_ptr<TensorImpl> OpenCLBackend::add(const TensorImpl* x1, const TensorImpl* x2)
{
	DType resType = solveBinaryOpDType(x1->dtype(), x2->dtype());
	return applyBinaryOp(x1, x2, "#define f(x1, x2) (x1+x2)", resType);
}

std::shared_ptr<TensorImpl> OpenCLBackend::subtract(const TensorImpl* x1, const TensorImpl* x2)
{
	DType resType = solveBinaryOpDType(x1->dtype(), x2->dtype());
	return applyBinaryOp(x1, x2, "#define f(x1, x2) (x1-x2)", resType);
}

std::shared_ptr<TensorImpl> OpenCLBackend::mul(const TensorImpl* x1, const TensorImpl* x2)
{
	DType resType = solveBinaryOpDType(x1->dtype(), x2->dtype());
	return applyBinaryOp(x1, x2, "#define f(x1, x2) (x1*x2)", resType);
}

std::shared_ptr<TensorImpl> OpenCLBackend::div(const TensorImpl* x1, const TensorImpl* x2)
{
	DType resType = solveBinaryOpDType(x1->dtype(), x2->dtype());
	return applyBinaryOp(x1, x2, "#define f(x1, x2) (x1/x2)", resType);
}

std::shared_ptr<TensorImpl> OpenCLBackend::equal(const TensorImpl* x1, const TensorImpl* x2)
{
	return applyBinaryOp(x1, x2, "#define f(x1, x2) (x1==x2)", DType::Bool);
}

std::shared_ptr<TensorImpl> OpenCLBackend::greater(const TensorImpl* x1, const TensorImpl* x2)
{
	return applyBinaryOp(x1, x2, "#define f(x1, x2) (x1>x2)", DType::Bool);
}

std::shared_ptr<TensorImpl> OpenCLBackend::lesser(const TensorImpl* x1, const TensorImpl* x2)
{
	return applyBinaryOp(x1, x2, "#define f(x1, x2) (x1<x2)", DType::Bool);
}

std::shared_ptr<TensorImpl> OpenCLBackend::logical_and(const TensorImpl* x1, const TensorImpl* x2)
{
	return applyBinaryOp(x1, x2, "#define f(x1, x2) (x1&&x2)", DType::Bool);
}

std::shared_ptr<TensorImpl> OpenCLBackend::logical_or(const TensorImpl* x1, const TensorImpl* x2)
{
	return applyBinaryOp(x1, x2, "#define f(x1, x2) (x1||x2)", DType::Bool);
}

std::shared_ptr<TensorImpl> OpenCLBackend::from(const TensorImpl* x)
{
	const void* ptr = x->data();
	if(ptr != nullptr)
		return createTensor(x->shape(), x->dtype(), ptr);

	OpenCLBackend* src_backend = dynamic_cast<OpenCLBackend*>(x->backend());
	if(src_backend != nullptr && src_backend->context()() == context()()) {
		auto buf = src_backend->copy(x);
		auto& buffer = reinterpret_cast<OpenCLBuffer*>(buf->buffer().get())->buffer();
		cl_int err = queue_.enqueueMigrateMemObjects({buffer}, CL_MIGRATE_MEM_OBJECT_HOST);
		if(err != CL_SUCCESS)
			throw EtError("OpenCL data migration failed. Error: " + std::to_string(err));
		return createTensor(x->shape(), x->dtype(), buffer);
	}

	void* buffer = malloc(x->size()*dtypeToSize(x->dtype()));
	x->backend()->copyToHost(x, buffer);
	auto res = createTensor(x->shape(), x->dtype(), buffer);
	free(buffer);
	return res;
}
