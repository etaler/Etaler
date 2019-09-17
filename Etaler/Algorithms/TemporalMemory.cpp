#include "TemporalMemory.hpp"

using namespace et;

TemporalMemory::TemporalMemory(const Shape& input_shape, size_t cells_per_column, size_t max_synapses_per_cell, Backend* backend)
	:input_shape_(input_shape)
{
	Shape connection_shape = input_shape + cells_per_column + max_synapses_per_cell;

	connections_ = constant(connection_shape, -1, backend);
	permanences_ = Tensor(connection_shape, DType::Float, backend);
}

std::pair<Tensor, Tensor> TemporalMemory::compute(const Tensor& x, const Tensor& last_state)
{
	et_assert(x.shape() == input_shape_);
	Tensor active_cells;
	if(last_state.has_value() == true)
		active_cells = burst(x, last_state);
	else
		active_cells = burst(x, zeros(x.shape()+cellsPerColumn(), DType::Bool, x.backend()));
	Tensor activity = cellActivity(active_cells, connections_, permanences_, connected_permanence_, active_threshold_);
	Tensor predictive_cells = cast(activity, DType::Bool);

	return {predictive_cells, active_cells};

}

void TemporalMemory::learn(const Tensor& active_cells, const Tensor& last_active)
{
	Tensor learning_cells = reverseBurst(active_cells);

	learnCorrilation(last_active, learning_cells, connections_, permanences_, permanence_inc_, permanence_dec_);
	growSynapses(last_active, learning_cells, connections_, permanences_, initial_permanence_);

}

void TemporalMemory::loadState(const StateDict& states)
{
	permanence_inc_ = std::any_cast<float>(states.at("permanence_inc"));
	permanence_dec_ = std::any_cast<float>(states.at("permanence_dec"));
	connected_permanence_ = std::any_cast<float>(states.at("connected_permanence"));
	active_threshold_ = std::any_cast<int>(states.at("active_threshold"));
	input_shape_ = std::any_cast<Shape>(states.at("input_shape"));
	connections_ = std::any_cast<Tensor>(states.at("connections"));
	permanences_ = std::any_cast<Tensor>(states.at("permanences"));
}

TemporalMemory TemporalMemory::to(Backend* b) const
{
	TemporalMemory tm = *this;
	tm.connections_ = connections_.to(b);
	tm.permanences_ = permanences_.to(b);

	return tm;
}
