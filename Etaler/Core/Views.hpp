#pragma once

#include "SmallVector.hpp"
#include "Shape.hpp"
#include "TensorImpl.hpp"

#include <variant>
#include <memory>

namespace et
{

struct RawView {};

struct ReshapeView
{
	Shape new_shape;
};

using TensorView = std::variant<RawView, ReshapeView>;

struct ViewTensor : public TensorImpl
{
	ViewTensor(std::shared_ptr<TensorImpl> parent, Shape shape, TensorView view)
		: TensorImpl(std::move(parent->backend_ptr())), parent_(std::move(parent)), view_(std::move(view))
		{
			dtype_ = parent_->dtype();
			shape_ = std::move(shape);
		}

	std::shared_ptr<TensorImpl> parent_;
	TensorView view_;
};

}