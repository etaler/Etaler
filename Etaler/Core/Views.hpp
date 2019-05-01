#pragma once

#include "SmallVector.hpp"
#include "Shape.hpp"
#include "TensorImpl.hpp"

#include <variant>
#include <memory>

namespace et
{

struct RawView {};

struct RectangularView
{
	RectangularView(svector<intmax_t> start, Shape shape)
		: start_(start), shape_(shape) {}

	RectangularView(Shape shape)
		: start_(svector<intmax_t>(shape.size(), 0)), shape_(shape) {}

	const svector<intmax_t>& start() const {return start_;}
	const Shape& shape() const {return shape_;}

	svector<intmax_t> start_;
	Shape shape_;
};

template <typename IdxType, typename ShapeType>
inline size_t unfoldIndex(const IdxType& index, const ShapeType& shape)
{
	size_t s = 0;
	size_t v = 1;
	assert(index.size() == shape.size());
	for(int i=(int)index.size()-1;i>=0;i--) {
		v *= (i==(int)index.size()-1?1:shape[i+1]);
		s += index[i] * v;
	}

	return s;
}

inline svector<intmax_t> shapeToStride(const Shape& shape)
{
	svector<intmax_t> v;
	v.resize(shape.size());
	size_t acc = 1;
	for(int i=(int)shape.size()-1;i>=0;i--) {
		acc *= shape[i];
		v[i] = acc;
	}
	return v;
}

template <typename ShapeType>
inline Shape foldIndex(size_t index, const ShapeType& shape)
{
	assert(shape.size() != 0);
	svector<intmax_t> v = shapeToStride(shape);
	Shape res;
	res.resize(v.size());
	for(size_t i=1;i<v.size();i++) {
		res[i-1] = index/v[i];
		index = index%v[i];
	}
	res.back() = index;
	return res;
}

using TensorView = std::variant<RawView, RectangularView>;

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

struct RangeIndex
{
	RangeIndex() = default;
	RangeIndex(intmax_t start)
	{
		start_ = start;
		end_ = start+1;
		count_from_back_ = false;
	}

	RangeIndex(intmax_t start, intmax_t end)
	{
		start_ = start;
		end_ = end;
		if (end < 0) {
			count_from_back_ = true;
			end = -end;
		}
		else
			count_from_back_ = false;
	}

	RangeIndex(intmax_t start, intmax_t end, bool count_from_back)
	{
		start_ = start;
		end_ = end;
		count_from_back_ = count_from_back;
	}

	intmax_t start() const {return start_;}
	intmax_t end() const {return end_;}
	bool countFromBack() const {return count_from_back_;}

	intmax_t start_ = 0;
	intmax_t end_ = 0;
	bool count_from_back_ = true;
};

inline RangeIndex all()
{
	return RangeIndex(0, 0, true);
}

inline RangeIndex range(intmax_t start, intmax_t end)
{
	return RangeIndex(start, end);
}

inline RangeIndex range(intmax_t end)
{
	return RangeIndex(0, end);
}


}