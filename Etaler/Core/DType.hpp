#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "Half.hpp"

namespace et
{

enum class DType
{
	Unknown = -1,
	Bool = 0,
	Int32,
	Float,
	Half,

	//Aliases
	Float32 = Float,
	Float16 = Half,
	Int = Int32,
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
	else if constexpr(std::is_same<T, float16>::value)
		return DType::Half;
	else
		return DType::Unknown;
}

inline constexpr size_t dtypeToSize(DType dtype)
{
	if(dtype == DType::Bool)
		return sizeof(bool);
	else if(dtype == DType::Int32)
		return sizeof(int32_t);
	else if(dtype == DType::Float)
		return sizeof(float);
	else if(dtype == DType::Half)
		return sizeof(float16);
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
	else if(dtype == DType::Half)
		return "half";
	return "Unknown";
}

inline std::string to_string(DType dtype)
{
	return to_ctype_string(dtype);
}

}