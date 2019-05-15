#pragma once

#include <cstdint>
#include <iostream>
#include <algorithm>

#include "SmallVector.hpp"

namespace et
{

const constexpr unsigned shapeSmallVecSize = 4;
class Shape : public svector<intmax_t, shapeSmallVecSize>
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

}
