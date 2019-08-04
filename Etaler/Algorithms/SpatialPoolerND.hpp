#pragma once

#include "SpatialPooler.hpp"

namespace et
{

//SpatialPooler with topology support
class SpatialPoolerND : public SpatialPooler
{
	SpatialPoolerND() = default;

	SpatialPoolerND(const Shape& input_shape, size_t kernel_size, size_t stride=1, float potential_pool_pct=0.75, size_t seed=42
		, float global_density = 0.15, float boost_factor = 0, Backend* b = defaultBackend());
};

}