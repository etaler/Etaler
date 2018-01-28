#pragma once

#include <memory>
#include <vector>
#include <variant>
#include <numeric>

#include "TensorImpl.hpp"
#include "Error.hpp"
#include "DefaultBackend.hpp"
#include "Views.hpp"
#include "TypeHelpers.hpp"

#include "Etaler_export.h"

namespace et
{

struct Tensor;
Tensor ETALER_EXPORT brodcast_to(const Tensor& t, Shape s);

ETALER_EXPORT std::ostream& operator<< (std::ostream& os, const Tensor& t);
std::string to_string(const Tensor& t);

struct ETALER_EXPORT Tensor
{
	Tensor() = default;
	Tensor(const Tensor&) = default;
	Tensor(std::shared_ptr<TensorImpl> pimpl)
		: pimpl_(std::move(pimpl)) {}
	explicit Tensor(Shape s, DType dtype, Backend* backend=defaultBackend())
		: pimpl_(backend->createTensor(s, dtype, nullptr)) {}
	template <typename T>
	Tensor(Shape s, const T* data, Backend* backend=defaultBackend())
	{
		constexpr DType dtype = typeToDType<T>();
		static_assert(std::is_same_v<T, void> == false);
		static_assert(dtype !=  DType::Unknown && "Cannot process this kind on data type");
		pimpl_ = backend->createTensor(s, dtype, data);
	}

	Tensor(int v) : Tensor({1}, &v) {}
	Tensor(float v) : Tensor({1}, &v) {}
	Tensor(bool v) : Tensor({1}, &v) {}

	Tensor& operator= (const Tensor& t)& { this->pimpl_ = t.pimpl_; return *this; } //l-value assignment. i.e. normal assignment
	Tensor& operator= (const Tensor& t)&& //r-value assignment. i.e. Assigning to a returned value
	{
		//The check here is for performance
		if(t.shape() != shape())
			assign(brodcast_to(t, shape()));
		else
			assign(t);
		return *this;
	}

	//Member and property access
	void* data() {return call_const(data);}
	const void* data() const {return pimpl_->data();}
	DType dtype() const {return pimpl_->dtype();}
	Shape shape() const {if(pimpl_) return pimpl_->shape(); else return Shape();}
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
		if(pimpl()->isplain() == false)
			return realize().toHost<T>();
		if(dtype() != typeToDType<T>()) {
			throw EtError("toHost() failed. Requested type and dtype mismatch. " + demangle(typeid(T).name())
				+ " requested but " + to_ctype_string(dtype()) + "is stored.");
		}
		std::vector<T> res(size());
		backend()->copyToHost(pimpl(), res.data());
		return res;
	}


	Tensor to(Backend* dest_backend) const;
	Tensor to(std::shared_ptr<Backend> dest_backend) const {return to(dest_backend.get());}
	Tensor copy() const;

	//View/Indexing
	Tensor view(svector<Range> ranges) const;

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
		if(pimpl() == source.pimpl())
			return;
		backend()->assign(pimpl(), source.pimpl());
	}

	Tensor realize() const
	{
		return backend()->realize(pimpl());
	}

	template <typename T>
	T item() const
	{
		if(size() != 1)
			throw EtError("item() can only be called on tensors with exactly 1 element");
		auto vec = toHost<T>();
		assert(vec.size() == 1);
		return vec[0];
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
	Tensor logical_not() const { return backend()->logical_not(pimpl()); }

	Tensor add(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->add(a(), b()); }
	Tensor subtract(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->subtract(a(), b()); }
	Tensor mul(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->mul(a(), b()); }
	Tensor div(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->div(a(), b()); }
	Tensor equal(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->equal(a(), b()); }
	Tensor greater(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->greater(a(), b()); }
	Tensor lesser(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->lesser(a(), b()); }
	Tensor logical_and(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->logical_and(a(), b()); }
	Tensor logical_or(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->logical_or(a(), b()); }

	Tensor operator- () const {return negate();}
	Tensor operator+ () const {return *this;}
	Tensor operator! () const {return logical_not();}

	Tensor operator+ (const Tensor& other) const {return add(other);}
	Tensor operator- (const Tensor& other) const {return subtract(other);}
	Tensor operator* (const Tensor& other) const {return mul(other);}
	Tensor operator/ (const Tensor& other) const {return div(other);}

	Tensor operator== (const Tensor& other) const {return equal(other);}
	Tensor operator< (const Tensor& other) const {return lesser(other);}
	Tensor operator> (const Tensor& other) const {return greater(other);}
	Tensor operator&& (const Tensor& other) const {return logical_and(other);}
	Tensor operator|| (const Tensor& other) const {return logical_or(other);}

	//Operators can the optimized
	Tensor operator<= (const Tensor& other) const {return lesser(other) || equal(other);}
	Tensor operator>= (const Tensor& other) const {return greater(other) || equal(other);}
	Tensor operator!= (const Tensor& other) const {return !equal(other);}

	//Subscription operator
	Tensor operator [] (svector<Range> r) { return view(r); }

	Tensor sum(intmax_t dim=-1, DType dtype=DType::Unknown) const;
	bool isSame (const Tensor& other) const;

	//Utils
	TensorImpl* operator () () {return pimpl();}
	const TensorImpl* operator () () const {return pimpl();}

	bool has_value() const {return (bool)pimpl_ && size() > 0;}

	std::pair<Tensor, Tensor> brodcast(const Tensor& other) const;

protected:
	std::shared_ptr<TensorImpl> pimpl_;
};

static Tensor operator+ (std::variant<float, int, bool, half> v, const Tensor& t)
{
	return std::visit([&t](auto v) {return Tensor(v)+t;}, v);
}

static Tensor operator- (std::variant<float, int, bool, half> v, const Tensor& t)
{
	return std::visit([&t](auto v) {return Tensor(v)-t;}, v);
}

static Tensor operator* (std::variant<float, int, bool, half> v, const Tensor& t)
{
	return std::visit([&t](auto v) {return Tensor(v)*t;}, v);
}

static Tensor operator/ (std::variant<float, int, bool, half> v, const Tensor& t)
{
	return std::visit([&t](auto v) {return Tensor(v)/t;}, v);
}

//Procedural  APIs
template <typename T>
Tensor constant(const Shape& shape, T value, Backend* backend=defaultBackend())
{
	static_assert(typeToDType<T>() != DType::Unknown);
	std::vector<T> v(shape.volume(), value);
	return Tensor(shape, v.data(), backend);
}

Tensor ETALER_EXPORT zeros(const Shape& shape, DType dtype=DType::Int32, Backend* backend=defaultBackend());
Tensor ETALER_EXPORT ones(const Shape& shape, DType dtype=DType::Int32, Backend* backend=defaultBackend());

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
	return x.backend()->cellActivity(x(), connections(), permeances(), connected_permeance, active_threshold, has_unconnected_synapse);
}

static void learnCorrilation(const Tensor& x, const Tensor& learn, const Tensor& connection
	, Tensor& permeances, float perm_inc, float perm_dec, bool has_unconnected_synapse=true)
{
	x.backend()->learnCorrilation(x(), learn(), connection(), permeances(), perm_inc, perm_dec, has_unconnected_synapse);
}

static Tensor globalInhibition(const Tensor& x, float fraction)
{
	return x.backend()->globalInhibition(x(), fraction);
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
	connection.backend()->sortSynapse(connection(), permeances());
}

static Tensor burst(const Tensor& x, const Tensor& s)
{
	return x.backend()->burst(x(), s());
}

static Tensor reverseBurst(const Tensor& x)
{
	return x.backend()->reverseBurst(x());
}

static void growSynapses(const Tensor& x, const Tensor& y, Tensor& connections, Tensor& permeances, float init_perm)
{
	x.backend()->growSynapses(x(), y(), connections(), permeances(), init_perm);
}

static void decaySynapses(Tensor& connections, Tensor& permeances, float threshold)
{
	connections.backend()->decaySynapses(connections(), permeances(), threshold);
}

static void assign(Tensor& x, const Tensor& y)
{
	x.assign(y);
}

Tensor ETALER_EXPORT sum(const Tensor& x, intmax_t dim=-1, DType dtype=DType::Unknown);
Tensor ETALER_EXPORT cat(const svector<Tensor>& tensors, intmax_t dim=0);
inline Tensor concat(const svector<Tensor>& tensors, intmax_t dim=0) { return cat(tensors, dim); }
inline Tensor concatenate(const svector<Tensor>& tensors, intmax_t dim=0) { return cat(tensors, dim); }
std::pair<Tensor, Tensor> brodcast_tensors(const Tensor& a, const Tensor& b);

static Tensor exp(const Tensor& x) { return x.exp(); }
static Tensor negate(const Tensor& x) { return x.negate(); }
static Tensor inverse(const Tensor& x) { return x.inverse(); }
static Tensor log(const Tensor& x) { return x.log(); }
static Tensor logical_not(const Tensor& x) { return x.logical_not(); }

static Tensor add(const Tensor& x1, const Tensor& x2) { return x1.add(x2); }
static Tensor subtract(const Tensor& x1, const Tensor& x2) { return x1.subtract(x2); }
static Tensor mul(const Tensor& x1, const Tensor& x2) { return x1.mul(x2); }
static Tensor div(const Tensor& x1, const Tensor& x2) { return x1.div(x2); }
static Tensor equal(const Tensor& x1, const Tensor& x2) { return x1.equal(x2); }
static Tensor greater(const Tensor& x1, const Tensor& x2) { return x1.greater(x2); }
static Tensor lesser(const Tensor& x1, const Tensor& x2) { return x1.lesser(x2); }
static Tensor logical_and(const Tensor& x1, const Tensor& x2) { return x1.logical_and(x2); }
static Tensor logical_or(const Tensor& x1, const Tensor& x2) { return x1.logical_or(x2); }

static Tensor zeros_like(const Tensor& x) { return zeros(x.shape(), x.dtype(), x.backend()); }
static Tensor ones_like(const Tensor& x) { return ones(x.shape(), x.dtype(), x.backend()); }
}

#include <sstream>

namespace cling
{

//FIXME: For some weard reson, I can't just return et::to_string(*value) and this function have to be inlined.
//Otherwise cling crashes.
inline std::string printValue(const et::Tensor* value)
{
	std::stringstream ss;
	ss << *value;
	return ss.str();
}

}
