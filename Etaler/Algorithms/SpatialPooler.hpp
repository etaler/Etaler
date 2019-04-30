#pragma once

#include <vector>

#include <random>

#include "Etaler/Core/Shape.hpp"
#include "Etaler/Core/Backend.hpp"
#include "Etaler/Core/Error.hpp"
#include "Etaler/Core/Tensor.hpp"
#include "Etaler/Core/Serialize.hpp"
#include "Etaler/Core/DefaultBackend.hpp"

namespace et
{

//TODO: Add topoligy, boosting support
struct SpatialPooler
{
	SpatialPooler() = default;
	SpatialPooler(const Shape& input_shape, const Shape& output_shape, float potential_pool_pct=0.75, size_t seed=42
		, float global_density = 0.15, Backend* b = defaultBackend());

	Tensor compute(const Tensor& x) const;

	void learn(const Tensor& x, const Tensor& y)
	{
		backend_->learnCorrilation(x.pimpl(), y.pimpl(), connections_.pimpl(), permances_.pimpl(), permance_inc_, permance_dec_);
	}

	void setPermanceInc(float inc) { permance_inc_ = inc; }
	float permanceInc() const {return permance_inc_;}

	void setPermanceDec(float dec) { permance_dec_ = dec; }
	float permanceDec() const {return permance_dec_;}

	void setConnectedPermance(float thr) { connected_permance_ = thr; }
	float connectedPermance() const { return connected_permance_; }

	void setActiveThreshold(size_t thr) { active_threshold_ = thr; }
	size_t activeThreshold() const { return active_threshold_; }

	void setGlobalDensity(float d) { global_density_ = d; }
	size_t globalDensity() const { return global_density_; }

	StateDict states() const
	{
		return {{"input_shape", input_shape_}, {"output_shape", output_shape_}, {"connections", connections_}
			, {"permances", permances_}, {"permance_inc", permance_inc_}, {"permance_dec", permance_dec_}
			, {"connected_permance", connected_permance_}, {"active_threshold", (int)active_threshold_}
			, {"global_density", global_density_}};
	}

	void loadState(const StateDict& states);
protected:
	float permance_inc_ = 0.1;
	float permance_dec_ = 0.1;
	float connected_permance_ = 0.21;
	size_t active_threshold_ = 5;
	float global_density_ = 0.1;

	Shape input_shape_;
	Shape output_shape_;
	Tensor connections_;
	Tensor permances_;
	Backend* backend_;
};


}
