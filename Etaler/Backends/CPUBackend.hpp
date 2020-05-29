#pragma once

#include <Etaler/Core/TensorImpl.hpp>
#include <Etaler/Core/TypeHelpers.hpp>

#include <memory>
#include <variant>
#include <vector>
#include <cstdint>


namespace et
{

struct ETALER_EXPORT CPUBuffer : public BufferImpl
{
	CPUBuffer(const Shape& shape, DType dtype, std::shared_ptr<Backend> backend)
		: BufferImpl(shape.volume(), dtype, std::move(backend))
	{
		if(dtype == DType::Bool)
			storage_ = new bool[shape.volume()];
		else if(dtype == DType::Int32)
			storage_ = new int32_t[shape.volume()];
		else if(dtype == DType::Float)
			storage_ = new float[shape.volume()];
		else if(dtype == DType::Half)
			storage_ = new half[shape.volume()];
		else
			std::cerr << "Critical Warning: CPUBuffer Initialize failed. Unknown DType" << std::endl;
	}

	CPUBuffer(const Shape& shape, DType dtype, std::shared_ptr<Backend> backend, const void* src_ptr)
		: CPUBuffer(shape, dtype, std::move(backend))
	{
		void* ptr = data();

		//HACK: Lazy method to dopy data
		if(src_ptr != nullptr)
			memcpy(ptr, src_ptr, shape.volume()*dtypeToSize(dtype));
	}

	virtual ~CPUBuffer();

	virtual void* data() const override;

protected:
	std::variant<bool*, int32_t*, float*, half*> storage_;
};

struct ETALER_EXPORT CPUBackend : public Backend
{
	virtual std::shared_ptr<TensorImpl> createTensor(const Shape& shape, DType dtype, const void* data=nullptr) override
	{
		auto buf = std::make_shared<CPUBuffer>(shape, dtype, shared_from_this(), data);
		return std::make_shared<TensorImpl>(buf, shape, shapeToStride(shape));
	}

	virtual std::shared_ptr<TensorImpl> cellActivity(const TensorImpl* x, const TensorImpl* connections, const TensorImpl* permeances,
		float connected_permeance, size_t active_threshold, bool has_unconnected_synapse=true) override;
	virtual void learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections,
		TensorImpl* permeances, float perm_inc, float perm_dec, bool has_unconnected_synapse=true) override;
	virtual std::shared_ptr<TensorImpl> globalInhibition(const TensorImpl* x, float fraction) override;
	virtual std::shared_ptr<TensorImpl> cast(const TensorImpl* x, DType toType) override;
	virtual void copyToHost(const TensorImpl* pimpl, void* dest) override;
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

	//Unary Operations
	virtual std::shared_ptr<TensorImpl> abs(const TensorImpl* x) override;
	virtual std::shared_ptr<TensorImpl> exp(const TensorImpl* x) override;
	virtual std::shared_ptr<TensorImpl> negate(const TensorImpl* x) override;
	virtual std::shared_ptr<TensorImpl> inverse(const TensorImpl* x) override;
	virtual std::shared_ptr<TensorImpl> log(const TensorImpl* x) override;
	virtual std::shared_ptr<TensorImpl> logical_not(const TensorImpl* x) override;

	//Binary Operations
	virtual std::shared_ptr<TensorImpl> add(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> subtract(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> mul(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> div(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> equal(const TensorImpl* x1, const TensorImpl* x2) override ;
	virtual std::shared_ptr<TensorImpl> greater(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> lesser(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> logical_and(const TensorImpl* x1, const TensorImpl* x2) override;
	virtual std::shared_ptr<TensorImpl> logical_or(const TensorImpl* x1, const TensorImpl* x2) override;

	virtual std::string name() const override {return "CPU";}
};

} // et
