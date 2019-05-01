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
	Tensor copy() const {return pimpl_->backend()->copy(pimpl_.get());}

	bool isSame (const Tensor& other) const;

	operator TensorImpl* () {return pimpl();}
	operator const TensorImpl* () const {return pimpl();}

	bool has_value() const {return (bool)pimpl_;}

	Tensor reshape(Shape shape) const
	{
		if(size() != (size_t)shape.volume())
			EtError("Cannot reshape from " + to_string(this->shape()) + " to " + to_string(shape));
		return std::make_shared<ViewTensor>(pimpl_, shape, RectangularView(shape));
	}

	Tensor view(svector<Range> ranges) const
	{
		if(ranges.size() > dimentions())
			throw EtError("Cannot view a tensor of " + std::to_string(dimentions()) + " with " + std::to_string(ranges.size()) + " dimentions");

		while(ranges.size() != dimentions())
			ranges.push_back(all());

		auto resolve_index = [](intmax_t idx, bool from_back, intmax_t size) {
			if(from_back == true)
				return size-idx;
			else
				return idx;
		};

		auto resolve_range_size = [&](Range r, intmax_t size) {
			return resolve_index(r.end(), r.endFromBack(), size) - resolve_index(r.start(), r.startFromBack(), size);
		};

		Shape view_shape;
		Shape result_shape;
		svector<intmax_t> offset;

		for(size_t i=0;i<dimentions();i++)  {
			Range r = ranges[i];

			offset.push_back(resolve_index(r.start(), r.startFromBack(), shape()[i]));
			intmax_t size = resolve_range_size(r, shape()[i]);

			et_assert(size > 0, "Negative steps not supported now");

			if(size != 1 || result_shape.size() != 0)
				result_shape.push_back(size);
		}
		if(result_shape.size() == 0)
			result_shape.push_back(1);

		view_shape = result_shape;
		if(view_shape.size() < dimentions()) {
			Shape s;
			s.resize(dimentions() - view_shape.size());
			for(auto v : view_shape)
				s.push_back(v);
			view_shape = s;
		}
		return std::make_shared<ViewTensor>(pimpl_, result_shape, RectangularView(offset, view_shape));
	}

protected:
	std::shared_ptr<TensorImpl> pimpl_;
};

std::ostream& operator<< (std::ostream& os, const Tensor& t);

//Healpers
inline Tensor createTensor(const Shape& shape, DType dtype=DType::Int32, void* data=nullptr)
{
	return defaultBackend()->createTensor(shape, dtype, data);
}

template <typename T>
inline Tensor createTensor(const Shape& shape, const T* data, Backend* backend)
{
	constexpr DType dtype = typeToDType<T>();
	static_assert(std::is_same_v<T, void> == false);
	static_assert(dtype ==  DType::Unknown && "Cannot process this kind on data type");
	return backend->createTensor(shape, dtype, data);
}

template <typename T>
inline Tensor createTensor(const Shape& shape, const T* data)
{
	return createTensor(shape, data, defaultBackend());
}

template <typename T>
Tensor constant(const Shape& shape, T value, Backend* backend=defaultBackend())
{
	static_assert(typeToDType<T>() != DType::Unknown);
	std::vector<T> v(shape.volume(), value);
	return backend->createTensor(shape, typeToDType<T>(), v.data());
}

Tensor zeros(const Shape& shape, DType dtype=DType::Int32, Backend* backend=defaultBackend());
Tensor ones(const Shape& shape, DType dtype=DType::Int32, Backend* backend=defaultBackend());

inline Tensor attempt_realize(const Tensor& t)
{
	if(points_to<const ViewTensor>(t.pimpl()) == false)
		return t;
	return t.backend()->realize(t);
}

}
