#include "TemporalMemory.hpp"

using namespace et;

TemporalMemory::TemporalMemory(const Shape& input_shape, size_t cells_per_column, size_t max_synapses_per_cell, Backend* backend)
		: cells_per_column_(cells_per_column), backend_(backend)
{
	Shape connection_shape = input_shape + cells_per_column + max_synapses_per_cell;

	connections_ = constant(connection_shape, -1, backend);
	permances_ = Tensor(connection_shape, DType::Float, backend);
}

std::pair<Tensor, Tensor> TemporalMemory::compute(const Tensor& x, const Tensor& last_state)
{
	Tensor active_cells;
	if(last_state.has_value() == true)
		active_cells = backend_->applyBurst(x, last_state);
	else
		active_cells = backend_->applyBurst(x, zeros(x.shape()+cells_per_column_, DType::Bool, backend_));
	Tensor overlap = backend_->overlapScore(active_cells, connections_, permances_, connected_permance_, active_threshold_);
	Tensor predictive_cells = backend_->cast(overlap, DType::Bool);

	return {predictive_cells, active_cells};

}

void TemporalMemory::learn(const Tensor& active_cells, const Tensor& last_active)
{
	if(last_active.has_value() == false)
		return;

	Tensor learning_cells = backend_->reverseBurst(active_cells);

	backend_->learnCorrilation(last_active, learning_cells, connections_, permances_, permance_inc_, permance_dec_);
	backend_->growSynapses(last_active, learning_cells, connections_, permances_, 0.21);

}