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
		: cells_per_column_(cells_per_column), backend_(backend)
	{
		Shape connection_shape = input_shape + cells_per_column + max_synapses_per_cell;
		std::vector<int32_t> conns(connection_shape.volume(), -1);

		connections_ = backend->createTensor(connection_shape, DType::Int32, conns.data());
		permances_ = backend->createTensor(connection_shape, DType::Float);
	}

	std::pair<Tensor, Tensor> compute(const Tensor& x, const Tensor& last_state)
	{
		Tensor active_cells;
		if(last_state.has_value() == true)
			active_cells = backend_->applyBurst(x, last_state);
		else
			active_cells = backend_->applyBurst(x, zeros(x.shape()+cells_per_column_, DType::Bool, backend_));
		Tensor overlap = backend_->overlapScore(active_cells, connections_, permances_, 0.1, 3);
		Tensor predictive_cells = backend_->cast(overlap, DType::Bool);
		return {predictive_cells, active_cells};

	}

	void learn(const Tensor& active_cells, const Tensor& last_active)
	{
		if(last_active.has_value() == false)
			return;

		Tensor learning_cells = backend_->reverseBurst(active_cells);

		backend_->learnCorrilation(last_active, learning_cells, connections_, permances_, 0.1, 0.1);
		backend_->growSynapses(last_active, learning_cells, connections_, permances_, 0.21);
	}

	size_t cells_per_column_;
	Tensor connections_;
	Tensor permances_;
	Backend* backend_;
};

}