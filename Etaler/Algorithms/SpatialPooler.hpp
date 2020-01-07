#pragma once

#include <vector>

#include <random>

#include "Etaler/Core/Shape.hpp"
#include "Etaler/Core/Backend.hpp"
#include "Etaler/Core/Error.hpp"
#include "Etaler/Core/Tensor.hpp"
#include "Etaler/Core/Serialize.hpp"
#include "Etaler/Core/DefaultBackend.hpp"
#include "Synapse.hpp"

#include "Etaler_export.h"

namespace et
{

struct ETALER_EXPORT SpatialPooler
{
	SpatialPooler() = default;
	SpatialPooler(const Shape& input_shape, const Shape& output_shape, float potential_pool_pct=0.75, size_t seed=42
		, float global_density = 0.15, float boost_factor = 0, Backend* b = defaultBackend());

	Tensor compute(const Tensor& x) const;

	void learn(const Tensor& x, const Tensor& y);

	void setPermanenceInc(float inc) { permanence_inc_ = inc; }
	float permanenceInc() const {return permanence_inc_;}

	void setPermanenceDec(float dec) { permanence_dec_ = dec; }
	float permanenceDec() const {return permanence_dec_;}

	void setConnectedPermanence(float thr) { connected_permanence_ = thr; }
	float connectedPermanence() const { return connected_permanence_; }

	void setActiveThreshold(size_t thr) { active_threshold_ = thr; }
	size_t activeThreshold() const { return active_threshold_; }

	void setGlobalDensity(float d) { global_density_ = d; }
	size_t globalDensity() const { return global_density_; }

	void setBoostingFactor(float f) { boost_factor_ = f; }
	float boostFactor() const { return boost_factor_; }

	Tensor connections() const {return connections_;}
	Tensor permanences() const {return permanences_;}

	StateDict states() const
	{
		return {{"input_shape", input_shape_}, {"output_shape", output_shape_}, {"connections", connections_}
			, {"permanences", permanences_}, {"permanence_inc", permanence_inc_}, {"permanence_dec", permanence_dec_}
			, {"connected_permanence", connected_permanence_}, {"active_threshold", (int)active_threshold_}
			, {"global_density", global_density_}, {"average_activity", average_activity_}
			, {"boost_factor", boost_factor_}};
	}



	SpatialPooler to(Backend* b) const;

	void loadState(const StateDict& states);

	SpatialPooler copy() const
	{
		return to(connections_.backend());
	}
//protected:
	float permanence_inc_ = 0.1;
	float permanence_dec_ = 0.1;
	float connected_permanence_ = 0.21;
	size_t active_threshold_ = 5;
	float global_density_ = 0.1;
	float boost_factor_ = 0;

	Shape input_shape_;
	Shape output_shape_;
	Tensor connections_;
	Tensor average_activity_;
	Tensor permanences_;
};


}
