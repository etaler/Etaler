#pragma once

#include <memory>
#include <vector>

#include "TensorImpl.hpp"
#include "Error.hpp"
#include "DefaultBackend.hpp"
#include "Views.hpp"
#include "TypeHelpers.hpp"

namespace et
{

struct Tensor;

std::ostream& operator<< (std::ostream& os, const Tensor& t);
std::string to_string(const Tensor& t);

struct Tensor
{
	Tensor() = default;
	Tensor(std::shared_ptr<TensorImpl> pimpl)
		: pimpl_(std::move(pimpl)) {}
	Tensor(Shape s, DType dtype, Backend* backend=defaultBackend())
		: pimpl_(backend->createTensor(s, dtype, nullptr)) {}
	template <typename T>
	Tensor(Shape s, const T* data, Backend* backend=defaultBackend())
	{
		constexpr DType dtype = typeToDType<T>();
		static_assert(std::is_same_v<T, void> == false);
		static_assert(dtype !=  DType::Unknown && "Cannot process this kind on data type");
		pimpl_ = backend->createTensor(s, dtype, data);
	}

	Tensor& operator= (const Tensor& t)& { this->pimpl_ = t.pimpl_; return *this; } //l-value assignment. i.e. normal assignment
	Tensor& operator= (const Tensor& t)&& { assign(t); return *this; } //r-value assignment. i.e. Assigning to a returned value

	//Member and property access
	void* data() {return call_const(data);}
	const void* data() const {return pimpl_->data();}
	DType dtype() const {return pimpl_->dtype();}
	Shape shape() const {return pimpl_->shape();}
	size_t size() const {return pimpl_->size();}
	size_t dimentions() const {return pimpl_->dimentions();}
	void resize(Shape s) {pimpl()->resize(s);}
	bool iscontiguous() const {return pimpl()->iscontiguous();}

	Backend* backend() const {return pimpl()->backend();};


	template <typename ImplType=TensorImpl>
	const ImplType* pimpl() const {return dynamic_cast<const ImplType*>(pimpl_.get());}

	template <typename ImplType=TensorImpl>
	TensorImpl* pimpl() {return call_const(pimpl<ImplType>);}

	std::shared_ptr<TensorImpl> shared_pimpl() const {return pimpl_;}

	//Data transfer
	template<typename T>
	std::vector<T> toHost() const
	{
		static_assert(std::is_same_v<T, bool> == false && "You should NOT use coptToHost<bool> as std::vector<bool> is a "
								"specillation. Use copyToHost<uint8_t> instead.");
		if(dtype() != typeToDType<T>())
			throw EtError("toHost() failed. Requested type and dtype mismatch");
		std::vector<T> res(size());
		backend()->copyToHost(pimpl(), res.data());
		return res;
	}


	Tensor to(Backend* dest_backend) const;
	Tensor to(std::shared_ptr<Backend> dest_backend) const {return to(dest_backend.get());}
	Tensor copy() const;

	//View/Indexing
	Tensor view(svector<Range> ranges) const;

	//TODO: Handle reshape for non-continous Tensor
	Tensor reshape(Shape shape) const
	{
		if(size() != (size_t)shape.volume())
			throw EtError("Cannot reshape from " + to_string(this->shape()) + " to " + to_string(shape));
		Tensor res = realize();
		res.resize(shape);
		return res;
	}

	Tensor flatten() const
	{
		Shape view_shape = {(intmax_t)size()};
		return reshape(view_shape);
	}

	Tensor swapaxis(size_t axis1, size_t axis2) const
	{
		et_assert(axis1 < dimentions());
		et_assert(axis2 < dimentions());
		Shape stride = pimpl_->stride();
		Shape s = shape();
		std::swap(stride[axis1], stride[axis2]);
		std::swap(s[axis1], s[axis2]);
		return std::make_shared<TensorImpl>(pimpl_->buffer(), s, stride, pimpl_->offset());
	}

	//Assigning and realizing
	void assign(const Tensor& source)
	{
		backend()->assign(*this, source);
	}

	Tensor realize() const
	{
		return backend()->realize(pimpl());
	}

	// Common Tensor operators
	Tensor cast(DType dtype) const
	{
		if(iscontiguous() == false)
			return realize().cast(dtype);
		return backend()->cast(pimpl(), dtype);
	}

	Tensor exp() const { return backend()->exp(pimpl()); }
	Tensor negate() const { return backend()->negate(pimpl()); }
	Tensor inverse() const { return backend()->inverse(pimpl()); }
	Tensor log() const { return backend()->log(pimpl()); }

	Tensor sum(intmax_t dim=-1, DType dtype=DType::Unknown) const;
	bool isSame (const Tensor& other) const;

	//Utils
	operator TensorImpl* () {return pimpl();}
	operator const TensorImpl* () const {return pimpl();}

	bool has_value() const {return (bool)pimpl_ && size() > 0;}

protected:
	std::shared_ptr<TensorImpl> pimpl_;
};

//Procedural  APIs
template <typename T>
Tensor constant(const Shape& shape, T value, Backend* backend=defaultBackend())
{
	static_assert(typeToDType<T>() != DType::Unknown);
	std::vector<T> v(shape.volume(), value);
	return Tensor(shape, v.data(), backend);
}

Tensor zeros(const Shape& shape, DType dtype=DType::Int32, Backend* backend=defaultBackend());
Tensor ones(const Shape& shape, DType dtype=DType::Int32, Backend* backend=defaultBackend());

inline Tensor realize(const Tensor& t)
{
	return t.realize();
}

inline Tensor attempt_realize(const Tensor& t)
{
	if(t.iscontiguous() == false)
		return t;
	return t.realize();
}

static Tensor cellActivity(const Tensor& x, const Tensor& connections, const Tensor& permeances
	, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse=true)
{
	return x.backend()->cellActivity(x, connections, permeances, connected_permeance, active_threshold, has_unconnected_synapse);
}

static void learnCorrilation(const Tensor& x, const Tensor& learn, const Tensor& connection
	, Tensor& permeances, float perm_inc, float perm_dec, bool has_unconnected_synapse=true)
{
	x.backend()->learnCorrilation(x, learn, connection, permeances, perm_inc, perm_dec, has_unconnected_synapse);
}

static Tensor globalInhibition(const Tensor& x, float fraction)
{
	return x.backend()->globalInhibition(x, fraction);
}

Tensor static cast(const Tensor& x, DType dtype)
{
	return x.cast(dtype);
}

static Tensor copy(const Tensor& x)
{
	return x.copy();
}

static void sortSynapse(Tensor& connection, Tensor& permeances)
{
	connection.backend()->sortSynapse(connection, permeances);
}

static Tensor burst(const Tensor& x, const Tensor& s)
{
	return x.backend()->burst(x, s);
}

static Tensor reverseBurst(const Tensor& x)
{
	return x.backend()->reverseBurst(x);
}

static void growSynapses(const Tensor& x, const Tensor& y, Tensor& connections, Tensor& permeances, float init_perm)
{
	x.backend()->growSynapses(x, y, connections, permeances, init_perm);
}

static void assign(Tensor& x, const Tensor& y)
{
	x.assign(y);
}

Tensor sum(const Tensor& x, intmax_t dim=-1, DType dtype=DType::Unknown);
std::pair<Tensor, Tensor> brodcast_tensors(const Tensor& a, const Tensor& b);

}
