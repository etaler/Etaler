#pragma once

#include <cstdint>
#include <iostream>
#include <algorithm>

#include "SmallVector.hpp"
#include "Error.hpp"

#include "Etaler_export.h"

namespace et
{

const constexpr unsigned shapeSmallVecSize = 4;
class ETALER_EXPORT Shape : public svector<intmax_t, shapeSmallVecSize>
{
public:
	const static intmax_t None = -1;
	Shape(std::initializer_list<intmax_t> l) : svector<intmax_t, shapeSmallVecSize>(l)
	{
	}

	template<class InputIt>
	Shape(InputIt first, InputIt last)
		: svector<intmax_t, shapeSmallVecSize>(first, last)
	{
	}

	Shape() : svector<intmax_t, shapeSmallVecSize>()
	{}

	Shape(size_t n, intmax_t init=0) : svector<intmax_t, shapeSmallVecSize>(n, init)
	{}

	void reserve(size_t n)
	{
		// only reserve when we can't fit the thing in the small_vector as reserve always allocates.
		if(n < shapeSmallVecSize)
			svector<intmax_t, shapeSmallVecSize>::reserve(n);
	}

	intmax_t volume() const
	{
		intmax_t val = 1;
		for(size_t i=0;i<size();i++)
			val *= operator[](i);
		return val;
	}

	bool operator ==(const Shape& other) const
	{
		if(size() != other.size())
			return false;
		for(size_t i=0;i<size();i++) {
			if(operator[](i) != other[i])
				return false;
		}
		return true;
	}

	inline bool match(const Shape& s) const
	{
		if(size() != s.size())
			return false;
		return std::equal(begin(), end(), s.begin(), [](auto a, auto b){return (a==None||b==None) || a==b;});
	}

	inline bool contains(intmax_t val) const
	{
		for(const auto& v : *this) {
			if(v == val)
				return true;
		}
		return false;
	}

	//The + operator pushes a value to the back
	inline Shape operator+ (value_type v) const
	{
		Shape s = *this;
		s.push_back(v);
		return s;
	}

	inline Shape operator+ (const Shape& rhs) const
	{
		Shape res = *this;
		for(auto v : rhs)
			res += v;
		return res;
	}

	inline void operator+= (value_type v)
	{
		push_back(v);
	}

	inline void operator+= (const Shape& rhs)
	{
		for(auto v : rhs)
			push_back(v);
	}

};

inline std::string to_string(const Shape& s)
{
	std::string res = "{";
	for(auto it=s.begin();it!=s.end();it++)
		res += (*it==Shape::None? "None" : std::to_string(*it)) + (it == s.end()-1 ? "" : ", ");
	res += "}";
	return res;
}

inline std::ostream& operator << (std::ostream& os, const Shape& s)
{
	os << to_string(s);
	return os;
}

inline Shape leftpad(Shape s, size_t size, intmax_t pad=1)
{
	if(s.size() >= size)
		return s;
	return Shape(size - s.size(), pad) + s;
}

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

static inline intmax_t convResultSize(intmax_t input, intmax_t kernel, intmax_t stride=1)
{
	return (input-kernel)/stride+1;
}

inline Shape convResultShape(const Shape& input, const Shape& kernel, const Shape& stride)
{
	assert(inptut.size() == kernel.size());
	assert(inptut.size() == stirde.size());

	Shape res(input.size());
	for(size_t i=0;i<input.size();i++) 
		res[i] = convResultSize(input[i], kernel[i], stride[i]);
	return res;
}

}

//Print Shape at the prompt
namespace cling
{

inline std::string printValue(const et::Shape* value)
{
	return et::to_string(*value);
}

}
