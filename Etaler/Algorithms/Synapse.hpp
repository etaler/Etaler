#pragma once

#include "Etaler/Core/Shape.hpp"
#include "Etaler/Core/Backend.hpp"
#include "Etaler/Core/Error.hpp"
#include "Etaler/Core/Tensor.hpp"
#include "Etaler/Core/DefaultBackend.hpp"

#include "Etaler_export.h"


namespace et::F
{
std::pair<Tensor, Tensor> ETALER_EXPORT gusianRandomSynapse(const Shape& input_shape, const Shape& output_shape, float potential_pool_pct=0.75
	, float mean = 0.21, float stddev = 1, size_t seed = 42, Backend* backend=defaultBackend());
std::pair<Tensor, Tensor> ETALER_EXPORT gusianRandomSynapseND(const Shape& input_shape, size_t kernel_size, size_t stride=1, float potential_pool_pct=0.75
	, float mean = 0.21, float stddev = 1, size_t seed = 42, Backend* backend=defaultBackend());
}