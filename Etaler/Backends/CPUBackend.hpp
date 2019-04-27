#pragma once

#include <Etaler/Core/TensorImpl.hpp>
#include <Etaler/Core/Tensor.hpp>

#include <memory>
#include <variant>
#include <vector>
#include <cstdint>


namespace et
{

struct CPUTensor : public TensorImpl
{
	CPUTensor(const Shape& shape, DType dtype, std::shared_ptr<Backend> backend)
		: TensorImpl(std::move(backend))
	{
		shape_ = shape;
		dtype_ = dtype;

		if(dtype == DType::Bool)
			storage_ = std::vector<uint8_t>(shape.volume());
		else if(dtype == DType::Int32)
			storage_ = std::vector<int32_t>(shape.volume());
		else if(dtype == DType::Float)
			storage_ = std::vector<float>(shape.volume());
		else
			std::cerr << "Critical Warning: CPUTensor Initialize failed. Unknown DType" << std::endl;
	}

	CPUTensor(const Shape& shape, DType dtype, std::shared_ptr<Backend> backend, const void* src_ptr)
		: CPUTensor(shape, dtype, std::move(backend))
	{
		void* ptr = data();

		//TODO: Lazy method to dopy data
		if(src_ptr != nullptr)
			memcpy(ptr, src_ptr, shape.volume()*dtypeToSize(dtype));
	}

	virtual const void* data() const override;
	virtual void* data() override {return call_const(data);}

protected:
	std::variant<std::vector<uint8_t>, std::vector<int32_t>, std::vector<float>> storage_;
};

struct CPUBackend : public Backend
{
	virtual std::shared_ptr<TensorImpl> createTensor(const Shape& shape, DType dtype, const void* data=nullptr) override
	{
		CPUTensor* ptr = new CPUTensor(shape, dtype, shared_from_this(), data);
		return std::shared_ptr<CPUTensor>(ptr, [this](TensorImpl* ptr){releaseTensor(ptr);});
	}

	virtual void releaseTensor(TensorImpl* pimpl) override
	{
		assert(dynamic_cast<CPUTensor*>(pimpl) != nullptr);
		delete pimpl;
	}

	virtual std::shared_ptr<TensorImpl> overlapScore(const TensorImpl* x, const TensorImpl* connections, const TensorImpl* permeances,
		float connected_permeance, size_t active_threshold, bool has_unconnected_synapse=true) override;
	virtual void learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections,
		TensorImpl* permeances, float perm_inc, float perm_dec) override;
	virtual std::shared_ptr<TensorImpl> globalInhibition(const TensorImpl* x, float fraction) override;
	virtual std::shared_ptr<TensorImpl> cast(const TensorImpl* x, DType toType) override;
	virtual void copyToHost(const TensorImpl* pimpl, void* dest) override;
	virtual std::shared_ptr<TensorImpl> copy(const TensorImpl* x) override;
	virtual void sortSynapse(TensorImpl* connections, TensorImpl* permeances) override;
	virtual std::shared_ptr<TensorImpl> applyBurst(const TensorImpl* x, const TensorImpl* s) override;
	virtual std::shared_ptr<TensorImpl> reverseBurst(const TensorImpl* x) override;
	virtual void growSynapses(const TensorImpl* x, const TensorImpl* y, TensorImpl* connections
		, TensorImpl* permeances, float initial_perm) override;

	virtual std::string name() const override {return "CPU";}
};

} // et
