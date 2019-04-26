#pragma once

#include <cstdint>
#include <limits>
#include <string>

namespace et
{

enum class DType
{
	Unknown = -1,
	Bool = 0,
	Int32,
	Float,
};

template <typename T>
constexpr inline DType typeToDType()
{
	if constexpr(std::is_same<T, float>::value)
		return DType::Float;
	else if constexpr(std::is_same<T, int32_t>::value)
		return DType::Int32;
	else if constexpr(std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value)
		return DType::Bool;
	else
		return DType::Unknown;
}

inline size_t dtypeToSize(DType dtype)
{
	if(dtype == DType::Bool)
		return sizeof(bool);
	else if(dtype == DType::Int32)
		return sizeof(int32_t);
	else if(dtype == DType::Float)
		return sizeof(float);
	return std::numeric_limits<size_t>::max();
}

inline std::string to_ctype_string(DType dtype)
{
	if(dtype == DType::Bool)
		return "bool";
	else if(dtype == DType::Int32)
		return "int";
	else if(dtype == DType::Float)
		return "float";
	return "Unknown";
}

}