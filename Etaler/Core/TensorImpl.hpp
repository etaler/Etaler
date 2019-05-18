#pragma once

#include "Shape.hpp"
#include "DType.hpp"
#include "Backend.hpp"

#include <memory>

namespace et
{

struct BufferImpl : public std::enable_shared_from_this<BufferImpl>
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

struct TensorImpl : public std::enable_shared_from_this<TensorImpl>
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

}
