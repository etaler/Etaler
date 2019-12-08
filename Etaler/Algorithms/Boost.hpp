#pragma once

#include <Etaler/Core/Tensor.hpp>

namespace et
{

static Tensor boostFactor(const Tensor& average_activity, float target_activity, float boost_factor)
{
        return exp((target_activity - average_activity) * boost_factor);
}

static Tensor boost(const Tensor& activity, const Tensor& average_activity, float target_activity, float boost_factor)
{
	if(boost_factor == 0)
		return activity.copy();
	return cast(boostFactor(average_activity, target_activity, boost_factor)*activity, DType::Int32);
}

static Tensor logarithmicBoost(const Tensor& activity, const Tensor& average_activity, float target_activity, float target_density, int active_threshold)
{
	return activity * (log(average_activity - target_activity)/log(target_density - sum(activity > active_threshold).item<int>()));
}

}