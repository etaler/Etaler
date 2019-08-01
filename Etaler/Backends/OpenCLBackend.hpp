#pragma once

#include <Etaler/Core/Backend.hpp>
#include <Etaler/Core/Error.hpp>
#include <Etaler/Core/TensorImpl.hpp>

//#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
	#include <OpenCL/cl.hpp>
#else
	#include <CL/cl.hpp>
#endif

#include <vector>
#include <deque>
#include <map>
#include <optional>

#include "Etaler_export.h"


namespace et
{


struct OpenCLBuffer : public BufferImpl
{
	OpenCLBuffer(const Shape& shape, DType dtype, const cl::Buffer& buffer, std::shared_ptr<Backend> backend)
		: BufferImpl(shape.volume(), dtype, backend), buffer_(buffer) {}

	cl::Buffer& buffer() {return buffer_;}
	const cl::Buffer& buffer() const {return buffer_;}

protected:
	cl::Buffer buffer_;
};

struct KernelManager
{
	KernelManager() : KernelManager(cl::Device(), cl::Context()) {};
	KernelManager(cl::Device device, cl::Context context);
	struct Application
	{
		cl::Program program;
		std::map<std::string, cl::Kernel> kernels;
	};

	cl::Kernel compileKernel(const std::string& src, const std::string& program_name, const std::string& kernel_name
		, bool force_override=false, const std::string& flags="");
	void compileKernel(const std::string& src, const std::string& program_name, const std::vector<std::string>& kernel_names
		, bool force_override=false, const std::string& flags="");
	void compileKernel(const std::vector<std::string>& srcs, const std::string& program_name, const std::vector<std::string>& kernel_names
		, bool force_override=false, const std::string& flags="");
	void compileFromFile(const std::string& paths, const std::string& program_name, const std::vector<std::string>& kernel_names
		, bool force_override=false, const std::string& flags="", const std::string& prepend="");
	void compileFromFile(const std::vector<std::string>& paths, const std::string& program_name, const std::vector<std::string>& kernel_names
		, bool force_override=false, const std::string& flags="", const std::string& prepend="");
	inline bool exists(const std::string& program_name, const std::string& kernel_name)
	{
		auto it = apps_.find(program_name);
		if(it == apps_.end())
			return false;
		if(it->second.kernels.find(kernel_name) == it->second.kernels.end())
			return false;
		return true;
	}
	const cl::Kernel& kernel(const std::string& program_name, const std::string& kernel_name) const {return apps_.at(program_name).kernels.at(kernel_name);}
	const cl::Kernel& kernel(const std::string& name) const {return kernel(name, name);}

	void addSearchPath(const std::string& path);

	void remove(std::string program_name) {auto it = apps_.find(program_name); if(it != apps_.end()) apps_.erase(it);}

	std::map<std::string, Application> apps_;
	cl::Device device_;
	cl::Context context_;
	std::map<std::string, std::string> kernelCache;

	std::deque<std::string> search_paths_ = {"./kernels/", "../kernels/", "/usr/local/share/Etaler/kernels/", "/usr/share/Etaler/kernels/"};

protected:
	std::string readKernel(const std::string& name);
};

struct ETALER_EXPORT OpenCLBackend : public Backend
{
	virtual ~OpenCLBackend() = default;
	OpenCLBackend();
	OpenCLBackend(size_t platform_id, size_t device_id);
	OpenCLBackend(cl::Context context, cl::Platform platform, cl::Device device);
	virtual std::shared_ptr<TensorImpl> createTensor(const Shape& shape, DType dtype, const void* data=nullptr) override;
	std::shared_ptr<TensorImpl> createTensor(const Shape& shape, DType dtype, cl::Buffer buf);
	void releaseTensor(OpenCLBuffer* pimpl);
	virtual void copyToHost(const TensorImpl* pimpl, void* dest) override;

	virtual void sync() const override;
	virtual std::string name() const override {return "OpenCL on " + device_.getInfo<CL_DEVICE_NAME>();}

	//Generats a string consisting the current device infomation for debug purpose
	std::string deviceInfo() const;

	virtual std::shared_ptr<TensorImpl> cellActivity(const TensorImpl* x, const TensorImpl* connections,
		const TensorImpl* permeances, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse=true) override;
	virtual void learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections,
		TensorImpl* permeances, float perm_inc, float perm_dec, bool has_unconnected_synapse=true) override;
	virtual std::shared_ptr<TensorImpl> globalInhibition(const TensorImpl* x, float fraction) override;
	virtual std::shared_ptr<TensorImpl> cast(const TensorImpl* x, DType toType) override;
	virtual std::shared_ptr<TensorImpl> copy(const TensorImpl* x) override;
	virtual void sortSynapse(TensorImpl* connections, TensorImpl* permeances) override;
	virtual std::shared_ptr<TensorImpl> burst(const TensorImpl* x, const TensorImpl* s) override;
	virtual std::shared_ptr<TensorImpl> reverseBurst(const TensorImpl* x) override;
	virtual void growSynapses(const TensorImpl* x, const TensorImpl* y, TensorImpl* connections
		, TensorImpl* permeances, float initial_perm) override;
	virtual void decaySynapses(TensorImpl* connections, TensorImpl* permeances, float threshold) override;
	virtual std::shared_ptr<TensorImpl> from(const TensorImpl* x) override;

	virtual std::shared_ptr<TensorImpl> realize(const TensorImpl* x) override;
	virtual void assign(TensorImpl* dest, const TensorImpl* src) override;
	virtual std::shared_ptr<TensorImpl> sum(const TensorImpl* x, size_t chunk_size, DType dtype=DType::Unknown) override;

	virtual std::shared_ptr<TensorImpl> exp(const TensorImpl* x) override;
	virtual std::shared_ptr<TensorImpl> negate(const TensorImpl* x) override;
	virtual std::shared_ptr<TensorImpl> inverse(const TensorImpl* x) override;
	virtual std::shared_ptr<TensorImpl> log(const TensorImpl* x) override;
	virtual std::shared_ptr<TensorImpl> logical_not(const TensorImpl* x) override;

	virtual std::shared_ptr<TensorImpl> add(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> subtract(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> mul(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> div(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> equal(const TensorImpl* x1, const TensorImpl* x2) override ;
	virtual std::shared_ptr<TensorImpl> greater(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> lesser(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> logical_and(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> logical_or(const TensorImpl* x1, const TensorImpl* x2) override;

	std::optional<cl::Buffer> toSparse(const TensorImpl* x);

	inline cl::Context context() {return context_;}

	inline bool isExtentionSupported(std::string ext) const
	{
		return (std::find(supported_extentions_.begin(), supported_extentions_.end(), ext)
			!= supported_extentions_.end());
	}

protected:

	void init(cl::Context context, cl::Platform platform, cl::Device device);

	cl_ulong localMemorySize() const
	{
		return local_mem_size_;
	}
	cl_device_local_mem_type localMemoryType() const
	{
		return local_mem_type_;
	}

	cl_uint numComputeUnits() const
	{
		return num_compute_units_;
	}

	inline cl::Buffer allocBuffer(size_t size)
	{
		cl_int err;
		cl::Buffer buf(context_, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, size, nullptr, &err);
		if(err != CL_SUCCESS)
			throw EtError("OpenCL memory allocation failed. Requested size: " + std::to_string(size) + ", " + " error: " + std::to_string(err));
		return buf;
	}

	std::shared_ptr<TensorImpl> applyUnaryOp(const TensorImpl* x, std::string f, DType resType);
	std::shared_ptr<TensorImpl> applyBinaryOp(const TensorImpl* x1, const TensorImpl* x2, std::string f, DType resType);


	KernelManager kernel_manager_;

	cl::Platform platform_;
	cl::Device device_;
	cl::Context context_;
	cl::CommandQueue queue_;

	cl_device_local_mem_type local_mem_type_;
	cl_ulong local_mem_size_;
	cl_uint num_compute_units_;

	std::vector<std::string> supported_extentions_;
	bool have_fp16_ = false;
};

}
