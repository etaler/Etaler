#pragma once

#include <cstdint>
#include <Etaler/3rdparty/half_precision/half.hpp>

#include <sstream>

namespace et
{

using half = half_precision::half;

using float16 = half;

}

namespace cling
{

inline std::string printValue(const et::half* value)
{
	std::stringstream ss;
	ss << (float)(*value);
	return ss.str();
}

}