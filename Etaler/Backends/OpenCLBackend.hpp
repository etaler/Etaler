#pragma once

#include <Etaler/Core/Backend.hpp>
#include <Etaler/Core/Error.hpp>
#include <Etaler/Core/TensorImpl.hpp>

//#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <vector>
#include <map>
#include <fstream>


namespace et
{


struct OpenCLTensor : public TensorImpl
{
	OpenCLTensor(const Shape& shape, DType dtype, const cl::Buffer& buffer, std::shared_ptr<Backend> backend)
		: TensorImpl(std::move(backend))
	{
		shape_ = shape;
		dtype_ = dtype;
		buffer_ = buffer;
	}

	cl::Buffer& buffer() {return buffer_;}
	const cl::Buffer& buffer() const {return buffer_;}

protected:
	cl::Buffer buffer_;
};

struct KernelManager
{
	KernelManager() = default;
	KernelManager(cl::Device device, cl::Context context)
		: device_(device), context_(context)
	{
	}

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
		, bool force_override=false, const std::string& flags="");
	void compileFromFile(const std::vector<std::string>& paths, const std::string& program_name, const std::vector<std::string>& kernel_names
		, bool force_override=false, const std::string& flags="");
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

	std::map<std::string, Application> apps_;
	cl::Device device_;
	cl::Context context_;
};

struct OpenCLBackend : public Backend
{
	virtual ~OpenCLBackend() = default;
	OpenCLBackend();
	virtual std::shared_ptr<TensorImpl> createTensor(const Shape& shape, DType dtype, const void* data=nullptr) override;
	virtual void releaseTensor(TensorImpl* pimpl) override;
	virtual void copyToHost(const TensorImpl* pimpl, void* dest) override;

	virtual void sync() const override;
	virtual std::string name() const override {return "OpenCL on " + device_.getInfo<CL_DEVICE_NAME>();}

	//Generats a string consisting the current device infomation for debug purpose
	std::string deviceInfo() const;

	virtual std::shared_ptr<TensorImpl> overlapScore(const TensorImpl* x, const TensorImpl* connections,
		const TensorImpl* permeances, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse=true) override;
	virtual void learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections,
		TensorImpl* permeances, float perm_inc, float perm_dec, bool has_unconnected_synapse=true) override;
	virtual std::shared_ptr<TensorImpl> globalInhibition(const TensorImpl* x, float fraction) override;
	virtual std::shared_ptr<TensorImpl> cast(const TensorImpl* x, DType toType) override;
	virtual std::shared_ptr<TensorImpl> copy(const TensorImpl* x) override;
	virtual void sortSynapse(TensorImpl* connections, TensorImpl* permeances) override;
	virtual std::shared_ptr<TensorImpl> applyBurst(const TensorImpl* x, const TensorImpl* s) override;
	virtual std::shared_ptr<TensorImpl> reverseBurst(const TensorImpl* x) override;
	virtual void growSynapses(const TensorImpl* x, const TensorImpl* y, TensorImpl* connections
		, TensorImpl* permeances, float initial_perm) override;

protected:

	cl_ulong localMemorySize() {return device_.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();}
	cl_device_local_mem_type localMemoryType() {return device_.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>();}

	inline cl::Buffer allocBuffer(size_t size)
	{
		cl_int err;
		cl::Buffer buf(context_, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, size, nullptr, &err);
		if(err != CL_SUCCESS)
			throw EtError("OpenCL memory allocation failed. Requested size: " + std::to_string(size) + ", " + " error: " + std::to_string(err));
		return buf;
	}

	KernelManager kernel_manager_;

	cl::Platform platform_;
	cl::Device device_;
	cl::Context context_;
	cl::CommandQueue queue_;

	std::string kernel_root_ = "../kernels/";
};

}