#pragma once

#include "Shape.hpp"
#include "DType.hpp"
#include "Backend.hpp"

#include <memory>

namespace et
{

struct TensorImpl : public std::enable_shared_from_this<TensorImpl>
{
	TensorImpl(std::shared_ptr<Backend> backend)
		: backend_(std::move(backend)) {}

	virtual ~TensorImpl() = default;
	virtual void* data() {return nullptr;}
	virtual const void* data() const {return nullptr;}

	DType dtype() const {return dtype_;}
	Shape shape() const {return shape_;}
	size_t dimentions() const {return shape_.size();}
	size_t size() const {return shape_.volume();}
	void resize(Shape s) {if(s.volume() == shape_.volume()) shape_ = s; else throw EtError("Cannot resize");}
	Backend* backend() const {return backend_.get();}
	std::shared_ptr<Backend> backend_ptr() const {return backend_;}

protected:
	DType dtype_ = DType::Unknown;
	Shape shape_;
	std::shared_ptr<Backend> backend_;

};

}
