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
	RectangularView(size_t offset, Shape strides)
		: offset_(offset), strides_(strides) {}

	RectangularView(Shape strides)
		: offset_(0), strides_(strides) {}

	size_t offset() const {return offset_;}
	const Shape& strides() const {return strides_;}

	size_t offset_;
	Shape strides_;
};

template <typename IdxType, typename StrideType>
inline size_t unfold(const IdxType& index, const StrideType& stride)
{
	size_t s = 0;
	et_assert(index.size() == stride.size());
	for(int i=(int)index.size()-1;i>=0;i--)
		s += stride[i] * index[i];

	return s;
}

inline Shape shapeToStride(const Shape& shape)
{
	Shape v;
	v.resize(shape.size());
	size_t acc = 1;
	v.back() = 1;
	for(int i=(int)shape.size()-1;i>0;i--) {
		acc *= shape[i];
		v[i-1] = acc;
	}
	return v;
}

template <typename IdxType, typename ShapeType>
inline size_t unfoldIndex(const IdxType& index, const ShapeType& shape)
{
	return unfold(index, shapeToStride(shape));
}

template <typename ShapeType>
inline Shape foldIndex(size_t index, const ShapeType& shape)
{
	assert(shape.size() != 0);
	svector<intmax_t> v = shapeToStride(shape);
	Shape res;
	res.resize(v.size());
	for(size_t i=0;i<v.size();i++) {
		res[i] = index/v[i];
		index = index%v[i];
	}

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

struct Range
{
	Range() = default;
	Range(intmax_t start)
	{
		start_ = start;
		end_ = start+1;
	}

	Range(intmax_t start, intmax_t end)
	{
		start_ = start;
		end_ = end;

		if (start < 0) {
			start_from_back_ = true;
			end = -end;
		}

		if (end < 0) {
			end_from_back_ = true;
			end = -end;
		}
	}

	Range(intmax_t start, intmax_t end, bool start_from_back, bool end_from_back)
	{
		start_ = start;
		et_assert(end >= 0);
		end_ = end;
		start_from_back_ = start_from_back;
		end_from_back_ = end_from_back;
	}

	intmax_t start() const {return start_;}
	intmax_t end() const {return end_;}
	bool startFromBack() const {return start_from_back_;}
	bool endFromBack() const {return end_from_back_;}

protected:
	intmax_t start_ = 0;
	bool start_from_back_ = false;
	intmax_t end_ = 0;
	bool end_from_back_ = false;
	//intmax_t step_size_ = 1;
};

inline Range all()
{
	return Range(0, 0, false, true);
}

inline Range range(intmax_t start, intmax_t end)
{
	return Range(start, end);
}

inline Range range(intmax_t end)
{
	return Range(0, end);
}


}
