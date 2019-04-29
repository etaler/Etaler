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
	TemporalMemory(const Shape& input_shape, size_t cells_per_column, size_t max_synapses_per_cell=64, Backend* backend=defaultBackend());
	std::pair<Tensor, Tensor> compute(const Tensor& x, const Tensor& last_state);
	void learn(const Tensor& active_cells, const Tensor& last_active);

	void setPermanceInc(float inc) { permance_inc_ = inc; }
	float permanceInc() const {return permance_inc_;}

	void setPermanceDec(float dec) { permance_dec_ = dec; }
	float permanceDec() const {return permance_dec_;}

	void setConnectedPermance(float thr) { connected_permance_ = thr; }
	float connectedPermance() const { return connected_permance_; }

	void setActiveThreshold(size_t thr) { active_threshold_ = thr; }
	size_t activeThreshold() const { return active_threshold_; }

	size_t cells_per_column_;
	float connected_permance_ = 0.1;
	size_t active_threshold_ = 2;
	float permance_inc_ = 0.1;
	float permance_dec_ = 0.1;
	Tensor connections_;
	Tensor permances_;
	Backend* backend_;
};

}
