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

template <typename T>
struct ETALER_EXPORT TensorIterator
{
	// Iterator properties
	using iterator_category = std::random_access_iterator_tag;
	using value_type = T;
	using raw_value_type = std::remove_const_t<value_type>; // extra
	using difference_type = intmax_t;
	using pointer = std::unique_ptr<raw_value_type>;
	using reference = T&;

	using ThisIterator = TensorIterator<T>;
	TensorIterator() = default;
	TensorIterator(reference t, intmax_t offset = 0) : t_(t), offset_(offset)
	{static_assert(std::is_same_v<raw_value_type, Tensor>); }
	value_type operator*() { return t_.view({offset_}); }
	// Unfortunatelly returning a pointer is not doable
	pointer operator->() { return std::make_unique<raw_value_type>(this->operator*()); }
	bool operator==(const ThisIterator& rhs) const { return offset_ == rhs.offset_ && t_.pimpl() == rhs.t_.pimpl(); }
	bool operator!=(const ThisIterator& rhs) const { return !(*this == rhs); }
	ThisIterator& operator++() {offset_ += 1; return *this;}
	ThisIterator operator++(int) {ThisIterator retval = *this; ++(*this); return retval;}
	ThisIterator& operator--() {offset_ -= 1; return *this;}
	ThisIterator operator--(int) {ThisIterator retval = *this; --(*this); return retval;}
	difference_type operator- (const ThisIterator& rhs) const { return offset_ - rhs.offset_; }
	ThisIterator operator+(intmax_t n) {return ThisIterator(t_,offset_+n);}
	ThisIterator operator-(intmax_t n) {return ThisIterator(t_,offset_-n);}
	ThisIterator& operator+=(intmax_t n) { offset_+=n; return *this; }
	ThisIterator& operator-=(intmax_t n) { offset_-=n; return *this; }
	value_type operator[](intmax_t n) { return *operator+(n); }
	bool operator< (const ThisIterator& rhs) const { return offset_ < rhs.offset_; }
	bool operator> (const ThisIterator& rhs) const { return offset_ > rhs.offset_; }
	bool operator<= (const ThisIterator& rhs) const { return offset_ <= rhs.offset_; }
	bool operator>= (const ThisIterator& rhs) const { return offset_ >= rhs.offset_; }
	value_type t_;
	intmax_t offset_ = 0;
};


Tensor ETALER_EXPORT brodcast_to(const Tensor& t, Shape s);

ETALER_EXPORT std::ostream& operator<< (std::ostream& os, const Tensor& t);
ETALER_EXPORT std::string to_string(const Tensor& t);

using IndexList = svector<std::variant<Range, intmax_t, int, size_t, unsigned int>>;

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

	template<typename T>
	Tensor(const std::vector<T>& vec, Backend* backend=defaultBackend())
		: Tensor(Shape{intmax_t(vec.size())}, vec.data()) {}

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
	size_t dimensions() const {return pimpl_->dimensions();}
	void resize(Shape s) {pimpl()->resize(s);}
	bool iscontiguous() const {return pimpl()->iscontiguous();}
	bool isplain() const {return pimpl()->isplain();}
	Shape stride() const {return pimpl()->stride();}

	Backend* backend() const {return pimpl()->backend();}


	template <typename ImplType=TensorImpl>
	const ImplType* pimpl() const {return dynamic_cast<const ImplType*>(pimpl_.get());}

	template <typename ImplType=TensorImpl>
	TensorImpl* pimpl() {return call_const(pimpl<ImplType>);}

	std::shared_ptr<TensorImpl> shared_pimpl() const {return pimpl_;}

	//Data transfer
	template<typename T>
	std::vector<T> toHost() const
	{
		if(pimpl()->isplain() == false)
			return realize().toHost<T>();
		if(dtype() != typeToDType<T>()) {
			throw EtError("toHost() failed. Requested type and dtype mismatch. " + demangle(typeid(T).name())
				+ " requested but " + to_ctype_string(dtype()) + " is stored.");
		}

		if constexpr(std::is_same_v<T, bool>) {
			std::unique_ptr<bool[]> buffer(new bool[size()]); // Incase copyToHost fails so we don't have memory leak
			backend()->copyToHost(pimpl(), buffer.get());
			std::vector<bool> res(buffer.get(), buffer.get()+size());
			return res;
		}
		else {
			std::vector<T> res(size());
			backend()->copyToHost(pimpl(), res.data());
			return res;
		}
	}


	Tensor to(Backend* dest_backend) const;
	Tensor to(const std::shared_ptr<Backend> dest_backend) const {return to(dest_backend.get());}
	Tensor copy() const;

	//View/Indexing
	Tensor view(const IndexList& ranges) const;

	Tensor reshape(Shape shape) const
	{
		if(std::find_if(shape.begin(), shape.end(), [](auto v){ return v < -1;}) != shape.end())
                        throw EtError("Sizes of each dimension should be gerater than one or is -1 (unknown). Got " + to_string(shape));
		size_t num_unknown = std::count_if(shape.begin(), shape.end(), [](auto v){return v == -1;}); // -1 indicates unknown size
		et_check(num_unknown <= 1, "Can only have 0 or 1 unknown dimensions. Got " + std::to_string(num_unknown));
		if(num_unknown == 1) {
			intmax_t known_volume = 1;
			size_t unknown_index = -1;
			for(size_t i=0;i<shape.size();i++) {
				if(shape[i] == -1)
					unknown_index = i;
				else
					known_volume *= shape[i];
			}
			assert(int(unknown_index) != -1);
			et_check(size() % known_volume == 0, "Cannot solve the unknown volume. Volume not divisible");
			shape[unknown_index] = size() / known_volume;
		}


		et_check(size() == (size_t)shape.volume(), "Cannot reshape from " + to_string(this->shape()) + " to " + to_string(shape));
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
		if(axis1 >= dimensions())
			throw EtError("Axis " + std::to_string(axis1) + " is out of range.");
		if(axis2 >= dimensions())
			throw EtError("Axis " + std::to_string(axis2) + " is out of range.");
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

	Tensor add(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->add(a.pimpl(), b.pimpl()); }
	Tensor subtract(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->subtract(a.pimpl(), b.pimpl()); }
	Tensor mul(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->mul(a.pimpl(), b.pimpl()); }
	Tensor div(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->div(a.pimpl(), b.pimpl()); }
	Tensor equal(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->equal(a.pimpl(), b.pimpl()); }
	Tensor greater(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->greater(a.pimpl(), b.pimpl()); }
	Tensor lesser(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->lesser(a.pimpl(), b.pimpl()); }
	Tensor logical_and(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->logical_and(a.pimpl(), b.pimpl()); }
	Tensor logical_or(const Tensor& other) const { auto [a, b] = brodcast(other); return backend()->logical_or(a.pimpl(), b.pimpl()); }

	inline bool any() const { return cast(DType::Bool).sum(std::nullopt, DType::Bool).item<uint8_t>(); }
	inline bool all() const { return cast(DType::Bool).sum(std::nullopt).item<int32_t>() == int32_t(size()); }

	// Neumeric operations
	Tensor operator+= (const Tensor& other) { *this = *this + other; return *this; }
	Tensor operator-= (const Tensor& other) { *this = *this - other; return *this; }
	Tensor operator*= (const Tensor& other) { *this = *this * other; return *this; }
	Tensor operator/= (const Tensor& other) { *this = *this / other; return *this; }

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
	Tensor operator [] (const IndexList& r) { return view(r); }
	template <typename ... Args>
	Tensor operator () (Args ... args) { return view({args ...}); }

	Tensor sum(std::optional<intmax_t> dim=std::nullopt, DType dtype=DType::Unknown) const;
	Tensor abs() const { return backend()->abs(pimpl()); }
	bool isSame (const Tensor& other) const;

	//Utils

	using iterator = TensorIterator<Tensor>;
	using const_iterator = TensorIterator<const Tensor>;

	iterator begin() { return iterator(*this, 0); }
	iterator back() { return iterator(*this, shape()[0]-1); }
	iterator end() { return iterator(*this, shape()[0]); }

	const_iterator begin() const { return const_iterator(*this, 0); }
	const_iterator back() const { return const_iterator(*this, shape()[0]-1); }
	const_iterator end() const { return const_iterator(*this, shape()[0]); }

	bool has_value() const {return (bool)pimpl_ && size() > 0;}

	std::pair<Tensor, Tensor> brodcast(const Tensor& other) const;

protected:
	std::shared_ptr<TensorImpl> pimpl_;
};

template <typename ScalarType>
inline Tensor operator+ (ScalarType v, const Tensor& t)
{
	return Tensor(v)+t;
}

template <typename ScalarType>
inline Tensor operator- (ScalarType v, const Tensor& t)
{
	return Tensor(v)-t;
}

template <typename ScalarType>
inline Tensor operator* (ScalarType v, const Tensor& t)
{
	return Tensor(v)*t;
}

template <typename ScalarType>
inline Tensor operator/ (ScalarType v, const Tensor& t)
{
	return Tensor(v)/t;
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

inline Tensor ravel(const Tensor& t)
{
	if(t.isplain() == true)
		return t;
	return t.realize();
}

inline Tensor cellActivity(const Tensor& x, const Tensor& connections, const Tensor& permeances
	, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse=true)
{
	const Tensor& input = [&](){
		if(x.dtype() == DType::Bool)
			return x;
		return x.cast(DType::Bool);
	}();
	return x.backend()->cellActivity(input.pimpl(), connections.pimpl(), permeances.pimpl(), connected_permeance, active_threshold, has_unconnected_synapse);
}

inline void learnCorrilation(const Tensor& x, const Tensor& learn, const Tensor& connection
	, Tensor& permeances, float perm_inc, float perm_dec, bool has_unconnected_synapse=true)
{
	x.backend()->learnCorrilation(x.pimpl(), learn.pimpl(), connection.pimpl(), permeances.pimpl(), perm_inc, perm_dec, has_unconnected_synapse);
}

inline Tensor globalInhibition(const Tensor& x, float fraction)
{
	return x.backend()->globalInhibition(x.pimpl(), fraction);
}

Tensor inline cast(const Tensor& x, DType dtype)
{
	return x.cast(dtype);
}

inline Tensor copy(const Tensor& x)
{
	return x.copy();
}

inline void sortSynapse(Tensor& connection, Tensor& permeances)
{
	connection.backend()->sortSynapse(connection.pimpl(), permeances.pimpl());
}

inline Tensor burst(const Tensor& x, const Tensor& s)
{
	return x.backend()->burst(x.pimpl(), s.pimpl());
}

inline Tensor reverseBurst(const Tensor& x)
{
	return x.backend()->reverseBurst(x.pimpl());
}

inline void growSynapses(const Tensor& x, const Tensor& y, Tensor& connections, Tensor& permeances, float init_perm)
{
	x.backend()->growSynapses(x.pimpl(), y.pimpl(), connections.pimpl(), permeances.pimpl(), init_perm);
}

inline void decaySynapses(Tensor& connections, Tensor& permeances, float threshold)
{
	connections.backend()->decaySynapses(connections.pimpl(), permeances.pimpl(), threshold);
}

inline void assign(Tensor& x, const Tensor& y)
{
	x.assign(y);
}

inline void swap(Tensor x, Tensor y)
{
	Tensor tmp = ravel(x).copy();
	x.assign(y);
	y.assign(tmp);
}

Tensor ETALER_EXPORT sum(const Tensor& x, std::optional<intmax_t> dim=std::nullopt, DType dtype=DType::Unknown);
Tensor ETALER_EXPORT cat(const svector<Tensor>& tensors, intmax_t dim=0);
inline Tensor concat(const svector<Tensor>& tensors, intmax_t dim=0) { return cat(tensors, dim); }
inline Tensor concatenate(const svector<Tensor>& tensors, intmax_t dim=0) { return cat(tensors, dim); }
std::pair<Tensor, Tensor> brodcast_tensors(const Tensor& a, const Tensor& b);

inline Tensor abs(const Tensor& x) { return x.abs(); }
inline Tensor exp(const Tensor& x) { return x.exp(); }
inline Tensor negate(const Tensor& x) { return x.negate(); }
inline Tensor inverse(const Tensor& x) { return x.inverse(); }
inline Tensor log(const Tensor& x) { return x.log(); }
inline Tensor logical_not(const Tensor& x) { return x.logical_not(); }
inline Tensor isclose(const Tensor& x, const Tensor& y, float rtol=1e-5f, float atol=1e-8f) { return abs(x-y) <= (atol + rtol * abs(y)); }

inline Tensor add(const Tensor& x1, const Tensor& x2) { return x1.add(x2); }
inline Tensor subtract(const Tensor& x1, const Tensor& x2) { return x1.subtract(x2); }
inline Tensor mul(const Tensor& x1, const Tensor& x2) { return x1.mul(x2); }
inline Tensor div(const Tensor& x1, const Tensor& x2) { return x1.div(x2); }
inline Tensor equal(const Tensor& x1, const Tensor& x2) { return x1.equal(x2); }
inline Tensor greater(const Tensor& x1, const Tensor& x2) { return x1.greater(x2); }
inline Tensor lesser(const Tensor& x1, const Tensor& x2) { return x1.lesser(x2); }
inline Tensor logical_and(const Tensor& x1, const Tensor& x2) { return x1.logical_and(x2); }
inline Tensor logical_or(const Tensor& x1, const Tensor& x2) { return x1.logical_or(x2); }

inline bool all(const Tensor& t) { return t.all(); }
inline bool any(const Tensor& t) { return t.any(); }

template <typename ... Args>
inline Tensor view(const Tensor& t, Args... args) { return t.view({args...}); }
inline Tensor dynamic_view(const Tensor& t, const IndexList& indices) { return t.view(indices); }

inline Tensor zeros_like(const Tensor& x) { return zeros(x.shape(), x.dtype(), x.backend()); }
inline Tensor ones_like(const Tensor& x) { return ones(x.shape(), x.dtype(), x.backend()); }
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
