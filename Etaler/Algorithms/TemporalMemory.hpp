#pragma once

#include "Etaler/Core/Shape.hpp"
#include "Etaler/Core/Backend.hpp"
#include "Etaler/Core/Error.hpp"
#include "Etaler/Core/Tensor.hpp"
#include "Etaler/Core/Serealize.hpp"
#include "Etaler/Core/DefaultBackend.hpp"

namespace et
{

struct TemporalMemory
{
	TemporalMemory(const Shape& input_shape, size_t cells_per_column, size_t max_synapses_per_cell=64, Backend* backend=defaultBackend())
		: backend_(backend)
	{
		Shape connection_shape = input_shape + cells_per_column + max_synapses_per_cell;
		std::vector<int32_t> conns(connection_shape.size(), -1);

		connections_ = backend->createTensor(connection_shape, DType::Int32, conns.data());
		permances_ = backend->createTensor(connection_shape, DType::Float);
	}

	Tensor compute(const Tensor& x)
	{
	}

	Tensor connections_;
	Tensor permances_;
	Backend* backend_;
};

}