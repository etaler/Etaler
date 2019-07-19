#pragma once

#include <cstdint>

namespace et
{

//A dummp float16 structure
struct half
{
	uint16_t dummy_;
};

using float16 = half;

}