#pragma once
#include <cassert>

#include "Shape.hpp"
#include "DType.hpp"
#include "Backend.hpp"
#include "TypeHelpers.hpp"

#include <memory>
#include <numeric>

namespace et
{

struct ETALER_EXPORT BufferImpl : public std::enable_shared_from_this<BufferImpl>
{
	BufferImpl(size_t size, DType dtype, std::shared_ptr<Backend> backend)
		: size_(size), dtype_(dtype), backend_(backend) {}
	virtual ~BufferImpl() = default;
	size_t size() const {return size_;}
	virtual void* data() const {return nullptr;}
	std::shared_ptr<Backend> backend() const {return backend_;}
	DType dtype() const {return dtype_;}
	size_t size_;
	DType dtype_ = DType::Unknown;
	std::shared_ptr<Backend> backend_;
};

struct ETALER_EXPORT TensorImpl : public std::enable_shared_from_this<TensorImpl>
{
	TensorImpl(std::shared_ptr<BufferImpl> buffer, Shape shape, Shape stride, size_t offset=0)
		: buffer_(buffer), shape_(shape), stride_(stride), offset_(offset) {}

	virtual ~TensorImpl() = default;
	void* data() {return buffer_->data();}
	const void* data() const {return buffer_->data();}

	DType dtype() const {return buffer_->dtype();}
	Shape shape() const {return shape_;}
	Shape stride() const {return stride_;}
	std::shared_ptr<BufferImpl> buffer() const {return buffer_;}
	size_t dimentions() const {return shape_.size();}
	size_t size() const {return shape_.volume();}
	size_t offset() const {return offset_;}
	void resize(Shape s) {if(isplain() && s.volume() == shape_.volume()){ shape_ = s; stride_ = shapeToStride(s);} else throw EtError("Cannot resize");}
	Backend* backend() const {return buffer_->backend().get();}
	std::shared_ptr<Backend> backend_ptr() const {return buffer_->backend();}
	bool iscontiguous() const {return shapeToStride(shape_) == stride_;}
	bool isplain() const {return shapeToStride(shape_) == stride() && offset() == 0;}

protected:
	std::shared_ptr<BufferImpl> buffer_;
	Shape shape_;
	Shape stride_;
	size_t offset_;
};

struct IsContingous {};
struct IsPlain {};

template <typename Storage>
struct IsDType
{      
	Storage types;
};

template<typename _Tp, typename... _Up>
IsDType(_Tp, _Up...)
-> IsDType<std::array<std::enable_if_t<(std::is_same_v<_Tp, _Up> && ...), _Tp>,
	1 + sizeof...(_Up)>>;


template <typename T>
bool checkProperty(const TensorImpl* x, const T& value)
{
	if constexpr(std::is_base_of_v<Backend, std::remove_pointer_t<std::decay_t<T>>>)
		return x->backend() == value;
	else if constexpr(std::is_same_v<T, DType>)
		return x->dtype() == value;
	else if constexpr(std::is_same_v<T, IsContingous>)
		return x->iscontiguous();
	else if constexpr(std::is_same_v<T, IsPlain>)
		return x->iscontiguous();
	else if constexpr(std::is_same_v<T, Shape>)
		return x->shape() == value;
	else if constexpr(is_specialization<std::remove_pointer_t<std::decay_t<T>>, IsDType>::value)
		return (std::find(value.types.begin(), value.types.end(), x->dtype()) != value.types.end());
	else
		et_assert(false, "a non-supported value is passed into checkProperty");
	return false;
}

template <typename T>
void requireProperty(const TensorImpl* x, const T value, const std::string& line, const std::string& v_name)
{
	if(checkProperty(x, value) == true)
		return;
	
	//Otherwise assertion failed
	const std::string msg = line + " Tensor property requirment not match. Expecting " + v_name;
	if constexpr(std::is_base_of_v<Backend, std::remove_pointer_t<std::decay_t<T>>>)
		throw EtError(msg + ".backend() == " + value->name());
	else if constexpr(std::is_same_v<T, DType>)
		throw EtError(msg + ".dtype() == " + to_ctype_string(value));
	else if constexpr(std::is_same_v<T, IsContingous>)
		throw EtError(msg + ".iscontiguous() == true");
	else if constexpr(std::is_same_v<T, IsPlain>)
		throw EtError(msg + ".isplain() == true");
	else if constexpr(std::is_same_v<T, Shape>)
		throw EtError(msg + " is expected to have shape " + to_string(value));
	else if constexpr(is_specialization<std::remove_pointer_t<std::decay_t<T>>, IsDType>::value) {
		throw EtError(msg + ".dtype() is in {" + std::accumulate(value.types.begin(), value.types.end(), std::string()
			, [](auto v, auto a){return v + to_ctype_string(a) + ", ";}));
	}
}

template <typename ... Args>
bool checkProperties(const TensorImpl* x, Args... args)
{
	return (checkProperty(x, args) && ...);
}

template <typename ... Args>
void requirePropertiesInternal(const TensorImpl* x, const std::string& line, const std::string& v_name, Args... args)
{
	(requireProperty(x, args, line, v_name), ...);
}

}

#define requireProperties(x, ...) (requirePropertiesInternal(x, std::string(__FILE__)+":"+std::to_string(__LINE__)\
	+":"+std::string(__func__)+"():", #x, __VA_ARGS__))
