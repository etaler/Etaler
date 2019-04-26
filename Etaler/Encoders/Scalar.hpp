#pragma once

#include <vector>
#include <Etaler/Core/Tensor.hpp>
#include <Etaler/Core/Backend.hpp>

namespace et
{

namespace encoder
{

static Tensor scalar(float v, float left=0, float right=1, size_t cells=32, size_t active_cells=4, Backend* backend=defaultBackend())
{
	float range = right - left;
	v = std::max(std::min(v, right), left);
	float fract = (v-left)/range;
	size_t start = (cells-active_cells)*fract;
	size_t end = start + active_cells;

	std::vector<uint8_t> vec(cells);
	for(size_t i=start;i<end;i++)
		vec[i] = 1;
	return backend->createTensor({(intmax_t)cells}, DType::Bool, vec.data());
}

}

}
