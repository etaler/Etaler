#include "OpenCLBackend.hpp"

#include "Etaler/Core/Random.hpp"
#include "Etaler/Core/Views.hpp"

#include <map>
#include <sstream>

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

inline size_t selectWorkSize(size_t max, size_t mul_of, size_t size)
{
	auto round = [mul_of](auto v){return ((v/mul_of)*mul_of) + (v%mul_of == 0 ? 0 : mul_of);};
	return std::min((size_t)max, round(size));
}

inline std::string hash_string(const std::string& str)
{
	auto hash = std::hash<std::string>()(str);
	std::stringstream ss;
	ss << std::hex << hash;
	return ss.str();
}

void replaceAll(std::string& str, const std::string& from, const std::string& to) {
	if(from.empty())
		return;
	size_t start_pos = 0;
	while((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
	}
}


template <typename T>
std::string str(T&& s)
{
	return std::to_string(s);
}

OpenCLBackend::OpenCLBackend()
{
	std::vector<cl::Platform> platforms;
	cl_int err = cl::Platform::get(&platforms);
	if(err != CL_SUCCESS)
		throw EtError("Failed to get OpenCL platforms. Error: " + std::to_string(err));
	if(platforms.size() == 0)
		throw EtError("No OpenCL platform found.");
	auto& platform = platforms[0];

	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if(devices.size() == 0)
		throw EtError("No OpenCL device found in platorm " + platform.getInfo<CL_PLATFORM_NAME>());
	auto& device = devices[0];
	if(device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>() == CL_FALSE)
		throw EtError("Compiler for " + device.getInfo<CL_DEVICE_NAME>() + " is not avliable. (Devices like Altera/Xilinx FPGAs not supported"
			" in the OpenCL backend.)");

	platform_ = platform;
	device_ = device;
	context_ = cl::Context(device, nullptr, nullptr, nullptr, &err);
	if(err != CL_SUCCESS)
		throw EtError("Failed to create OpenCL context. Error " + std::to_string(err));

	//I trust these won't fail
	queue_ = cl::CommandQueue(context_);
	kernel_manager_ = KernelManager(device, context_);
	kernel_manager_.compileKernel("kernel void __etaler_dummy__(global int* p){p[get_global_id(0)] = 0;}", "__etaler_dummy__", "__etaler_dummy__");
}

OpenCLBackend::OpenCLBackend(cl::Context context, cl::Platform platform, cl::Device device)
{
	context_ = std::move(context);
	platform_ = std::move(platform);
	device_ = std::move(device);

	//I trust these won't fail
	queue_ = cl::CommandQueue(context_);
	kernel_manager_ = KernelManager(device, context_);
	kernel_manager_.compileKernel("kernel void __etaler_dummy__(global int* p){p[get_global_id(0)] = 0;}", "__etaler_dummy__", "__etaler_dummy__");
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

	OpenCLTensor* ptr = new OpenCLTensor(shape, dtype, buf, shared_from_this());
	return std::shared_ptr<OpenCLTensor>(ptr, [this](TensorImpl* ptr){releaseTensor(ptr);});
}

void OpenCLBackend::releaseTensor(TensorImpl* pimpl)
{
	et_assert(dynamic_cast<OpenCLTensor*>(pimpl) != nullptr);
	delete pimpl;
}

void OpenCLBackend::copyToHost(const TensorImpl* pimpl, void* dest)
{
	const OpenCLTensor* t = dynamic_cast<const OpenCLTensor*>(pimpl);
	if(t == nullptr)
		throw EtError("Cannot copy to host memory: Tensor/Backend mismach");
	cl_int err = queue_.enqueueReadBuffer(t->buffer(), CL_TRUE, 0, t->size()*dtypeToSize(t->dtype()), dest);
	if(err != CL_SUCCESS)
		throw EtError("OpenCL buffer readback failed. Error: " + std::to_string(err));
}

std::string OpenCLBackend::deviceInfo() const
{
	std::map<int, std::string> cache_type;
	cache_type[CL_LOCAL] = "Local";
	cache_type[CL_GLOBAL] = "Global";
	cache_type[CL_NONE] = "None";
	std::string res;
	res += "Platform: " + platform_.getInfo<CL_PLATFORM_NAME>() + "\n";
	res += "Device name: " + device_.getInfo<CL_DEVICE_NAME>() + "\n";
	res += "Global memory size: " + std::to_string((float)device_.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/1024/1024) + " GB\n";
	res += "Max allocatable memory: " + std::to_string((float)device_.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()/1024/1024) + " GB\n";
	res += "Local memory size: " + std::to_string(device_.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/1024) + " KB\n";
	res += "Prefered work group size: " + std::to_string(kernel_manager_.kernel("__etaler_dummy__").getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device_)) + "\n";
	return res;
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
	, bool force_override, const std::string& flags)
{
	compileFromFile(std::vector<std::string>{path}, program_name, kernel_names, force_override, flags);
}

void KernelManager::compileFromFile(const std::vector<std::string>& paths, const std::string& program_name, const std::vector<std::string>& kernel_names
	, bool force_override, const std::string& flags)
{
	std::vector<std::string> sources;
	for(const auto& path : paths)
		sources.emplace_back(readAll(path));
	compileKernel(sources, program_name, kernel_names, force_override, flags);
}

std::shared_ptr<TensorImpl> OpenCLBackend::overlapScore(const TensorImpl* x, const TensorImpl* connections,
	const TensorImpl* permeances, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse)
{
	et_assert(points_to<const OpenCLTensor>(x));
	et_assert(points_to<const OpenCLTensor>(connections));
	et_assert(points_to<const OpenCLTensor>(permeances));

	et_assert(x->dtype() == DType::Bool);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);
	et_assert(connections->shape() == permeances->shape());
	et_assert(connections->dimentions() >= 2);

	Shape s = connections->shape();
	s.pop_back();
	auto y = createTensor(s, DType::Int32);

	auto args = "-DINPUT_SIZE="+str(x->size())+" -DMAX_SYNAPSE_PER_CELL="+str(connections->shape().back())+" -DNO_UNUSED_SYNAPSE=" + str(!has_unconnected_synapse);
	auto hash = hash_string(args);
	auto program_name = "overlapScore"+hash;

	if(x->size() < localMemorySize())
		kernel_manager_.compileFromFile(kernel_root_+"overlapScore.cl", program_name, {"overlapScore"}, false, args);
	else
		kernel_manager_.compileFromFile(kernel_root_+"overlapScore_global.cl", program_name, {"overlapScore"}, false, args);
	cl::Kernel k = kernel_manager_.kernel(program_name, "overlapScore");

	k.setArg(0, reinterpret_cast<const OpenCLTensor*>(x)->buffer());
	k.setArg(1, reinterpret_cast<const OpenCLTensor*>(connections)->buffer());
	k.setArg(2, reinterpret_cast<const OpenCLTensor*>(permeances)->buffer());
	k.setArg(3, reinterpret_cast<OpenCLTensor*>(y.get())->buffer());
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
	et_assert(points_to<OpenCLTensor>(x));

	et_assert(x->dtype() == DType::Int32);

	auto y = createTensor(x->shape(), DType::Bool);

	auto args = "-DINPUT_SIZE="+str(x->size())+" -DMAX_INPUT_VALUE="+str(2000);
	auto hash = hash_string(args);
	auto program_name = "overlapScore"+hash;

	cl::Kernel topKKernel, thresholdKernel;
	kernel_manager_.compileFromFile(kernel_root_+"globalInhibition.cl", program_name, {"fastTopK", "threshold"}, false, args);

	topKKernel = kernel_manager_.kernel(program_name, "fastTopK");
	thresholdKernel = kernel_manager_.kernel(program_name, "threshold");

	cl::Buffer threshold = allocBuffer(sizeof(int));

	topKKernel.setArg(0, reinterpret_cast<const OpenCLTensor*>(x)->buffer());
	topKKernel.setArg(1, threshold);
	topKKernel.setArg(2, (int)(x->size()*fraction));

	queue_.enqueueNDRangeKernel(topKKernel, cl::NullRange, cl::NDRange(256), cl::NDRange(256));

	thresholdKernel.setArg(0, reinterpret_cast<const OpenCLTensor*>(x)->buffer());
	thresholdKernel.setArg(1, reinterpret_cast<const OpenCLTensor*>(y.get())->buffer());
	thresholdKernel.setArg(2, threshold);
	queue_.enqueueNDRangeKernel(thresholdKernel, cl::NullRange, cl::NDRange(1024), cl::NDRange(32));

	return y;
}

std::shared_ptr<TensorImpl> OpenCLBackend::cast(const TensorImpl* x, DType toType)
{
	et_assert(points_to<OpenCLTensor>(x));
	auto args = "-DInType="+to_ctype_string(x->dtype())+" -DOutType="+to_ctype_string(toType);
	auto hash = hash_string(args);
	auto program_name = "cast"+hash;
	kernel_manager_.compileFromFile(kernel_root_+"cast.cl", program_name, {"cast"}, false, args);
	cl::Kernel k = kernel_manager_.kernel(program_name, "cast");

	auto res = createTensor(x->shape(), toType);
	k.setArg(0, reinterpret_cast<const OpenCLTensor*>(x)->buffer());
	k.setArg(1, reinterpret_cast<OpenCLTensor*>(res.get())->buffer());
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
	et_assert(points_to<OpenCLTensor>(x));
	size_t buf_size = x->size()*dtypeToSize(x->dtype());
	cl::Buffer buf = allocBuffer(buf_size);
	const cl::Buffer& src = reinterpret_cast<const OpenCLTensor*>(x)->buffer();
	cl_int err = queue_.enqueueCopyBuffer(src, buf, 0, 0, buf_size);
	if(err != CL_SUCCESS)
		throw EtError("Data copy enqueuing failed. Error " + std::to_string(err));

	OpenCLTensor* ptr = new OpenCLTensor(x->shape(), x->dtype(), buf, shared_from_this());
	return std::shared_ptr<OpenCLTensor>(ptr, [this](TensorImpl* ptr){releaseTensor(ptr);});
}

void OpenCLBackend::learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections,
	TensorImpl* permeances, float perm_inc, float perm_dec, bool has_unconnected_synapse)
{
	et_assert(points_to<OpenCLTensor>(x));
	et_assert(points_to<OpenCLTensor>(connections));
	et_assert(points_to<OpenCLTensor>(permeances));
	et_assert(points_to<OpenCLTensor>(learn));

	et_assert(connections->shape() == permeances->shape());
	et_assert(x->shape() == learn->shape());
	et_assert(x->dtype() == DType::Bool);
	et_assert(learn->dtype() == DType::Bool);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);

	auto args = "-DINPUT_SIZE="+str(x->size())+" -DMAX_SYNAPSE_PER_CELL="+str(connections->shape().back())+" -DNO_UNUSED_SYNAPSE="+str(!has_unconnected_synapse)
		+" -DOUTPUT_SIZE="+str(learn->size());
	auto hash = hash_string(args);
	auto program_name = "learnCorrilation"+hash;

	if(x->size() < localMemorySize())
		kernel_manager_.compileFromFile(kernel_root_+"learnCorrilation.cl", program_name, {"learnCorrilation"}, false, args);
	else
		kernel_manager_.compileFromFile(kernel_root_+"learnCorrilation_global.cl", program_name, {"learnCorrilation"}, false, args);
	cl::Kernel k = kernel_manager_.kernel(program_name, "learnCorrilation");

	k.setArg(0, reinterpret_cast<const OpenCLTensor*>(x)->buffer());
	k.setArg(1, reinterpret_cast<const OpenCLTensor*>(learn)->buffer());
	k.setArg(2, reinterpret_cast<const OpenCLTensor*>(connections)->buffer());
	k.setArg(3, reinterpret_cast<OpenCLTensor*>(permeances)->buffer());
	k.setArg(4, (float)perm_inc);
	k.setArg(5, (float)perm_dec);

	size_t local_size = 128;

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, x->size())), cl::NDRange(local_size));

	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel learnCorrilation execution failed. Code " + str(err));
}

void OpenCLBackend::sortSynapse(TensorImpl* connections, TensorImpl* permeances)
{
	et_assert(connections->shape() == permeances->shape());
	et_assert(points_to<OpenCLTensor>(connections));
	et_assert(points_to<OpenCLTensor>(permeances));
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);

	auto args = "-DMAX_SYNAPSE_PER_CELL="+str(connections->shape().back());
	auto program_name = "sortSynapse"+hash_string(args);
	kernel_manager_.compileFromFile(kernel_root_+"sort.cl", program_name, {"sortSynapse"}, false, args);
	cl::Kernel k = kernel_manager_.kernel(program_name, "sortSynapse");

	int num_cells = connections->size()/connections->shape().back();

	cl::Buffer aux_buffer1 = allocBuffer(connections->size()*sizeof(int));
	cl::Buffer aux_buffer2 = allocBuffer(permeances->size()*sizeof(float));

	k.setArg(0, reinterpret_cast<const OpenCLTensor*>(connections)->buffer());
	k.setArg(1, reinterpret_cast<const OpenCLTensor*>(permeances)->buffer());
	k.setArg(2, num_cells);
	k.setArg(3, aux_buffer1);
	k.setArg(4, aux_buffer2);
	size_t local_size = 128;

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, num_cells)), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel execution failed. Code " + str(err));
}

std::shared_ptr<TensorImpl> OpenCLBackend::applyBurst(const TensorImpl* x, const TensorImpl* s)
{
	et_assert(points_to<const OpenCLTensor>(x));
	et_assert(points_to<const OpenCLTensor>(s));
	et_assert(x->dtype() == DType::Bool);
	et_assert(s->dtype() == DType::Bool);

	Shape shape = s->shape();
	shape.pop_back();
	et_assert(shape == x->shape());

	auto res = copy(s);

	size_t num_columns = shape.volume();

	auto args = "-DCELLS_PER_COLUMN="+str(s->shape().back())+" -DNUM_COLUMNS="+str(num_columns);
	auto program_name = "applyBurst"+hash_string(args);
	kernel_manager_.compileFromFile(kernel_root_+"applyBurst.cl", program_name, {"applyBurst"}, false, args);
	cl::Kernel k = kernel_manager_.kernel(program_name, "applyBurst");

	k.setArg(0, reinterpret_cast<const OpenCLTensor*>(x)->buffer());
	k.setArg(1, reinterpret_cast<OpenCLTensor*>(res.get())->buffer());

	size_t local_size = 128;
	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, num_columns)), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel applyBurst execution failed. Code " + str(err));
	return res;
}

std::shared_ptr<TensorImpl> OpenCLBackend::reverseBurst(const TensorImpl* x)
{
	et_assert(points_to<const OpenCLTensor>(x));
	et_assert(x->dtype() == DType::Bool);

	size_t cells_per_column = x->shape().back();
	size_t num_columns = x->size()/cells_per_column;
	static pcg32 rng; //Static so the behavor hangees every time, breaking symmetry
	std::uniform_int_distribution<size_t> dist(0, cells_per_column-1);

	auto res = copy(x);

	auto args = "-DCELLS_PER_COLUMN="+str(cells_per_column)+" -DNUM_COLUMNS="+str(num_columns);
	auto program_name = "reverseBurst"+hash_string(args);
	kernel_manager_.compileFromFile(kernel_root_+"reverseBurst.cl", program_name, {"reverseBurst"}, false, args);
	cl::Kernel k = kernel_manager_.kernel(program_name, "reverseBurst");

	k.setArg(0, reinterpret_cast<const OpenCLTensor*>(res.get())->buffer());
	k.setArg(1, rng());
	k.setArg(2, rng());

	size_t local_size = 128;
	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, num_columns)), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel reverseBurst execution failed. Code " + str(err));
	return res;
}

void OpenCLBackend::growSynapses(const TensorImpl* x, const TensorImpl* y, TensorImpl* connections
		, TensorImpl* permeances, float initial_perm)
{
	et_assert(points_to<const OpenCLTensor>(x));
	et_assert(points_to<const OpenCLTensor>(y));
	et_assert(points_to<OpenCLTensor>(connections));
	et_assert(points_to<OpenCLTensor>(permeances));

	et_assert(x->dtype() == DType::Bool);
	et_assert(y->dtype() == DType::Bool);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);

	et_assert(x->shape() == y->shape());
	et_assert(connections->shape() == permeances->shape());
	Shape s = connections->shape();
	s.pop_back();
	et_assert(s == y->shape());

	size_t max_synapses_per_cell = connections->shape().back();
	size_t input_cell_count = x->size();

	auto args = "-DNUM_CELLS="+str(y->size())+" -DNUM_INPUT_BITS="+str(x->size())+" -DMAX_SYNAPSE_PER_CELL="+str(max_synapses_per_cell);
	auto program_name = "growSynapses"+hash_string(args);
	kernel_manager_.compileFromFile(kernel_root_+"growSynapses.cl", program_name, {"growSynapses"}, false, args);
	cl::Kernel k = kernel_manager_.kernel(program_name, "growSynapses");

	size_t local_size = 32;
	size_t work_size = selectWorkSize(4096, local_size, y->size());
	size_t num_groups = work_size/local_size;

	static cl::Buffer aux = allocBuffer(input_cell_count*num_groups);
	if(input_cell_count*num_groups > aux.getInfo<CL_MEM_SIZE>())
		aux = allocBuffer(input_cell_count*num_groups);
	cl::Buffer sparse_x = toSparse(x);

	k.setArg(0, sparse_x);
	k.setArg(1, reinterpret_cast<const OpenCLTensor*>(y)->buffer());
	k.setArg(2, reinterpret_cast<OpenCLTensor*>(connections)->buffer());
	k.setArg(3, reinterpret_cast<OpenCLTensor*>(permeances)->buffer());
	k.setArg(4, initial_perm);
	k.setArg(5, (int)(sparse_x.getInfo<CL_MEM_SIZE>()/sizeof(int)));
	k.setArg(6, aux);

	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(work_size), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel growSynapses execution failed. Code " + str(err));
}

cl::Buffer OpenCLBackend::toSparse(const TensorImpl* x)
{
	et_assert(points_to<const OpenCLTensor>(x));
	et_assert(x->dtype() == DType::Bool);

	auto args = "-DINPUT_SIZE="+str(x->size());
	auto program_name = "toSparse"+hash_string(args);
	kernel_manager_.compileFromFile(kernel_root_+"toSparse.cl", program_name, {"toSparse", "onBits"}, false, args);

	cl::Kernel on_bits = kernel_manager_.kernel(program_name, "onBits");
	cl::Kernel k = kernel_manager_.kernel(program_name, "toSparse");

	cl::Buffer num = allocBuffer(sizeof(int));

	on_bits.setArg(0, reinterpret_cast<const OpenCLTensor*>(x)->buffer());
	on_bits.setArg(1, num);
	cl_int err = queue_.enqueueNDRangeKernel(on_bits, cl::NullRange, cl::NDRange(256), cl::NDRange(256));

	int* num_on = (int*) queue_.enqueueMapBuffer(num, CL_TRUE, CL_MAP_READ, 0, sizeof(int));
	cl::Buffer buf = allocBuffer((*num_on)*sizeof(int));

	queue_.enqueueUnmapMemObject(num, num_on);

	k.setArg(0, reinterpret_cast<const OpenCLTensor*>(x)->buffer());
	k.setArg(1, buf);
	err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(256), cl::NDRange(256));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel toSparse execution failed. Code " + str(err));
	return buf;
}

static std::string jitRawView(const TensorImpl* x, size_t id)
{
	std::string func = R"(
int location_func$ID(int location)
{
	return location;
}
)";
	replaceAll(func, "$ID", std::to_string(id));
	return func;
}

static std::string jitDataAccess(const OpenCLTensor* x, size_t id)
{
	return jitRawView(x, id);
}

static std::string jitRectangularView(const OpenCLTensor* x, const RectangularView& view, size_t id)
{
	std::string func = R"(
int location_func$ID(int location)
{
	int in_stride[] = $IN_STRIDE;
	int bias[] = $BIAS;
	int stride[] = $STRIDE;

	int ndpos[$DIMS];
	int loc = location;
	#pragma unroll
	for(int i=0;i<$DIMS;i++)
		ndpos[i] = 0;
	#pragma unroll
	for(int i=0;i<$IN_DIMS;i++) {
		int stride = (i==$IN_DIMS-1 ? 1 : in_stride[i+1]);
		ndpos[$DIMS-$IN_DIMS+i] = loc/stride;
		loc %= stride;
	}

	#pragma unroll
	for(int i=0;i<$IN_DIMS;i++)
		ndpos[$DIMS-$IN_DIMS+i] += bias[i];

	int sum = 0;
	#pragma unroll
	for(int i=0;i<$DIMS;i++)
		sum += ndpos[i]*(i==$DIMS-1 ? 1 : stride[i+1]);
	return sum;
}
)";
	const TensorImpl* parent = reinterpret_cast<const ViewTensor*>(x)->parent_.get();
	replaceAll(func, "$ID", std::to_string(id));
	auto in_strides = shapeToStride(x->shape());
	Shape s = Shape(in_strides.begin(), in_strides.end());
	replaceAll(func, "$IN_STRIDE", to_string(s));
	replaceAll(func, "$IN_DIMS", std::to_string(x->dimentions()));
	replaceAll(func, "$DIMS", std::to_string(parent->dimentions()));

	auto strides = shapeToStride(parent->shape());
	replaceAll(func, "$STRIDE", to_string(Shape(strides.begin(), strides.end())));
	replaceAll(func, "$BIAS", to_string(Shape(view.start().begin(), view.start().end())));
	return func;
}

static std::string jitConversionFunc(const TensorImpl* t)
{
	std::string func = R"(
int calc_location(int i)
{
	int location0 = i;
	$CONV
	return position;
}
)";

	std::string conv;
	TensorImpl const* x = t;
	for(size_t i=0;;i++) {
		if(points_to<ViewTensor>(x) == false) {
			std::string call = "int position = location_func$ID(location$ID);\n";
			replaceAll(call, "$ID", std::to_string(i));
			conv += call;
			break;
		}
		else {
			std::string call = "int location$NEXT = location_func$ID(location$ID);\n";
			replaceAll(call, "$ID", std::to_string(i));
			replaceAll(call, "$NEXT", std::to_string(i+1));
			conv += call;
		}
		x = reinterpret_cast<const ViewTensor*>(x)->parent_.get();
	}

	replaceAll(func, "$CONV", conv);
	return func;
}

static std::string jitCopyFromViewKernel(const TensorImpl* t)
{
	std::string func = R"(
#define Type $TYPE

kernel void copy(global Type* restrict x, global Type* restrict y)
{
	int global_id = get_global_id(0);
	int global_size = get_global_size(0);
	for(int i=global_id;i<$SIZE;i+=global_size) {
		int position = calc_location(i);
		y[i] = x[position];
	}
}
)";

	auto s = shapeToStride(t->shape());

	std::string type = to_ctype_string(t->dtype());
	replaceAll(func, "$TYPE", type);
	replaceAll(func, "$SIZE", std::to_string(t->size()));

	return func;
}

static std::string jitCopyToViewKernel(const TensorImpl* t)
{
	std::string func = R"(
#define Type $TYPE

kernel void copy(global Type* restrict x, global Type* restrict y)
{
	int global_id = get_global_id(0);
	int global_size = get_global_size(0);
	for(int i=global_id;i<$SIZE;i+=global_size) {
		int position = calc_location(i);
		y[position] = x[i];
	}
}
)";

	auto s = shapeToStride(t->shape());

	std::string type = to_ctype_string(t->dtype());
	replaceAll(func, "$TYPE", type);
	replaceAll(func, "$SIZE", std::to_string(t->size()));

	return func;
}

static std::vector<std::string> jitPositionConvertionFuncs(const TensorImpl* x)
{
	std::vector<std::string> functions;
	const TensorImpl* t = x;

	//Generate all the functions need to convert the coordinates
	while(true) {
		if(points_to<const OpenCLTensor>(t) == true) {
			functions.push_back(jitDataAccess(reinterpret_cast<const OpenCLTensor*>(x), functions.size()));
			break;
		}

		et_assert(points_to<const ViewTensor>(t));
		std::visit([&](const auto& view) {
			using ViewType = std::decay_t<decltype(view)>;
			if constexpr(std::is_same_v<ViewType, RectangularView>)
				functions.push_back(jitRectangularView(reinterpret_cast<const OpenCLTensor*>(x), view, functions.size()));
			else if constexpr(std::is_same_v<ViewType, RawView>)
				functions.push_back(jitRawView(reinterpret_cast<const OpenCLTensor*>(x), functions.size()));
			else
				throw EtError("View "+ std::string(typeid(ViewType).name()) +"  supported");

		}, reinterpret_cast<const ViewTensor*>(t)->view_);

		t = reinterpret_cast<const ViewTensor*>(t)->parent_.get();
	}
	return functions;
}

static std::vector<std::string> jitCopyFromView(const TensorImpl* x)
{
	std::vector<std::string> functions = jitPositionConvertionFuncs(x);
	functions.push_back(jitConversionFunc(x));
	//Generate the kernel function
	functions.push_back(jitCopyFromViewKernel(x));
	return functions;
}

static std::vector<std::string> jitCopyToView(const TensorImpl* x)
{
	std::vector<std::string> functions = jitPositionConvertionFuncs(x);
	functions.push_back(jitConversionFunc(x));
	//Generate the kernel function
	functions.push_back(jitCopyToViewKernel(x));
	return functions;
}

static const OpenCLTensor* findSourceTensor(const TensorImpl* t)
{
	if(points_to<const OpenCLTensor>(t) == false)
		return findSourceTensor(reinterpret_cast<const ViewTensor*>(t)->parent_.get());
	return reinterpret_cast<const OpenCLTensor*>(t);
}

std::shared_ptr<TensorImpl> OpenCLBackend::realize(const TensorImpl* x)
{
	if(points_to<const OpenCLTensor>(x) == true)
		return copy(x);
	if(points_to<const ViewTensor>(x) == false)
		throw EtError("Cannot realize tensor, not a OpenCLTensor or ViewTensor");

	std::vector<std::string> conversion = jitCopyFromView(x);

	kernel_manager_.compileKernel(conversion, "__copy", {"copy"});
	cl::Kernel k = kernel_manager_.kernel("__copy", "copy");

	cl::Buffer buf = allocBuffer(x->size()*dtypeToSize(x->dtype()));
	auto source = findSourceTensor(x);
	k.setArg(0, source->buffer());
	k.setArg(1, buf);

	size_t local_size = 128;
	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, source->size())), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel execution failed. Code " + str(err));

	//for(auto s : conversion)
	//	std::cout << s << std::endl;

	kernel_manager_.remove("__copy");//We are unlikely to use this kernel again?

	return std::make_shared<OpenCLTensor>(x->shape(), x->dtype(), buf, shared_from_this());
}

void OpenCLBackend::assign(TensorImpl* dest, const TensorImpl* src)
{
	et_assert(points_to<OpenCLTensor>(dest) || points_to<ViewTensor>(dest));
	et_assert(points_to<OpenCLTensor>(src) || points_to<ViewTensor>(src));

	if(dest->shape() != src->shape())
	throw EtError("Shape mismatch in tensor assignment. Shape "
		+ to_string(dest->shape()) + "and " + to_string(src->shape()));

	auto source = realize(src);

	if(dest->dtype() != src->dtype())
		source = cast(realize(source.get()).get(), dest->dtype());

	std::vector<std::string> conversion = jitCopyToView(dest);

	kernel_manager_.compileKernel(conversion, "__copy", {"copy"});
	cl::Kernel k = kernel_manager_.kernel("__copy", "copy");

	auto desitnation = findSourceTensor(dest);
	k.setArg(0, reinterpret_cast<OpenCLTensor*>(source.get())->buffer());
	k.setArg(1, desitnation->buffer());

	size_t local_size = 128;
	cl_int err = queue_.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(selectWorkSize(4096, local_size, source->size())), cl::NDRange(local_size));
	if(err != CL_SUCCESS)
		throw EtError("OpenCL kernel execution failed. Code " + str(err));

	kernel_manager_.remove("__copy");//We are unlikely to use this kernel again?
}
