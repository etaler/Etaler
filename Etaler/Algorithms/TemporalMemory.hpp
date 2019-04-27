#pragma once

#include "Etaler/Core/Shape.hpp"
#include "Etaler/Core/Backend.hpp"
#include "Etaler/Core/Error.hpp"
#include "Etaler/Core/Tensor.hpp"
#include "Etaler/Core/Serialize.hpp"
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

	std::pair<Tensor, Tensor> compute(const Tensor& x, const Tensor& last_state)
	{
		Tensor active_cells = last_state.copy();
		backend_->applyBurst(x, active_cells);

	}

	void learn(const Tensor& active_cells, const Tensor& last_active)
	{
		Tensor learning_cells = last_active.copy();
		backend_->reverseBurst(learning_cells);

		backend_->learnCorrilation(active_cells, learning_cells, connections_, permances_, 0.1, 0.1);
		backend_->growSynapses(active_cells, learning_cells, connections_, permances_, 0.21);
	}

	Tensor connections_;
	Tensor permances_;
	Tensor predictive_cells_;
	Tensor active_cells_;
	Backend* backend_;
};

}