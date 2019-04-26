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
		cl::Kernel kernel;
	};

	cl::Kernel compileKernel(const std::string& src, const std::string& program_name, const std::string& kernel_name
		, bool force_override=false, const std::string& flag="");
	void compileKernel(const std::string& src, const std::string& program_name, const std::vector<std::string>& kernel_name
		, bool force_override=false, const std::string& flag="");;
	bool exists(const std::string& program_name, const std::string& kernel_name) {return kernels_.find(program_name+"."+kernel_name) != kernels_.end();}
	const cl::Kernel& kernel(const std::string& program_name, const std::string& kernel_name) const {return kernels_.at(program_name+"."+kernel_name).kernel;}
	const cl::Kernel& kernel(const std::string& name) const {return kernel(name, name);}

	std::map<std::string, Application> kernels_;
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

	virtual void overlapScore(const TensorImpl* x, const TensorImpl* connections,
		const TensorImpl* permeances, float connected_permeance, size_t active_threshold, TensorImpl* y, bool has_unconnected_synapse=true) override;
	virtual void learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections,
		TensorImpl* permeances, float perm_inc, float perm_dec) override;
	virtual void globalInhibition(const TensorImpl* x, TensorImpl* y, float fraction) override;
	virtual std::shared_ptr<TensorImpl> cast(const TensorImpl* x, DType toType) override;
	virtual std::shared_ptr<TensorImpl> copy(const TensorImpl* x) override;
	virtual void sortSynapse(TensorImpl* connections, TensorImpl* permeances) override;

protected:

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