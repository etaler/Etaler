#pragma once

#include <memory>
#include <vector>

#include "TensorImpl.hpp"
#include "Error.hpp"
#include "DefaultBackend.hpp"
#include "Views.hpp"

namespace et
{

//A handy macro to call duplicated const version of the non const function.
#define call_const(func) [&](){\
	auto r = const_cast<const std::remove_reference<decltype(*this)>::type*>(this)->func();\
	using T = decltype(r); \
	if constexpr(std::is_pointer<T>::value) \
		return const_cast<typename std::remove_const<typename std::remove_pointer<T>::type>::type*>(r);\
	else \
		return r;\
}()


struct Tensor
{
	Tensor() = default;
	Tensor(std::shared_ptr<TensorImpl> pimpl)
		: pimpl_(std::move(pimpl)) {}
	Tensor(std::shared_ptr<ViewTensor> pimpl)
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

	void* data() {return call_const(data);}
	const void* data() const {return pimpl_->data();}
	DType dtype() const {return pimpl_->dtype();}
	Shape shape() const {return pimpl_->shape();}
	size_t size() const {return pimpl_->size();}
	size_t dimentions() const {return pimpl_->dimentions();}
	void resize(Shape s) {pimpl()->resize(s);}

	Backend* backend() const {return pimpl()->backend();};


	template <typename ImplType=TensorImpl>
	const ImplType* pimpl() const {return dynamic_cast<const ImplType*>(pimpl_.get());}

	template <typename ImplType=TensorImpl>
	TensorImpl* pimpl() {return call_const(pimpl<ImplType>);}

	template<typename T>
	std::vector<T> toHost() const
	{
		static_assert(std::is_same_v<T, bool> == false && "You should NOT use coptToHost<bool> as std::vector<bool> is a "
								"specillation. Use copyToHost<uint8_t> instead.");
		et_assert(dtype() == typeToDType<T>());
		std::vector<T> res(size());
		backend()->copyToHost(pimpl(), res.data());
		return res;
	}


	Tensor to(Backend* dest_backend) const;
	Tensor to(std::shared_ptr<Backend> dest_backend) const {return to(dest_backend.get());}
	Tensor copy() const {return backend()->copy(pimpl_.get());}

	bool isSame (const Tensor& other) const;

	operator TensorImpl* () {return pimpl();}
	operator const TensorImpl* () const {return pimpl();}

	bool has_value() const {return (bool)pimpl_;}
	Tensor view(svector<Range> ranges) const;

	Tensor reshape(Shape shape) const
	{
		if(size() != (size_t)shape.volume())
			throw EtError("Cannot reshape from " + to_string(this->shape()) + " to " + to_string(shape));
		return std::make_shared<ViewTensor>(pimpl_, shape, RawView());
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
		Shape stride = shapeToStride(shape());
		Shape s = shape();
		std::swap(stride[axis1], stride[axis2]);
		std::swap(s[axis1], s[axis2]);
		return std::make_shared<ViewTensor>(pimpl_, s, stride);
	}

	void assign(const Tensor& source)
	{
		backend()->assign(*this, source);
	}

	Tensor realize() const
	{
		return backend()->realize(pimpl());
	}

	Tensor cast(DType dtype) const
	{
		return backend()->cast(pimpl(), dtype);
	}

protected:
	std::shared_ptr<TensorImpl> pimpl_;
};

std::ostream& operator<< (std::ostream& os, const Tensor& t);
std::string to_string(const Tensor& t);

//Healpers
template <typename T>
Tensor constant(const Shape& shape, T value, Backend* backend=defaultBackend())
{
	static_assert(typeToDType<T>() != DType::Unknown);
	std::vector<T> v(shape.volume(), value);
	return Tensor(shape, v.data());
}

Tensor zeros(const Shape& shape, DType dtype=DType::Int32, Backend* backend=defaultBackend());
Tensor ones(const Shape& shape, DType dtype=DType::Int32, Backend* backend=defaultBackend());

inline Tensor realize(const Tensor& t)
{
	if(points_to<const ViewTensor>(t.pimpl()) == false)
		return t;
	return t.backend()->realize(t);
}

//Functional APIs
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
	return x.backend()->cast(x, dtype);
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

}
