#pragma once

#include <Etaler/Core/Tensor.hpp>

namespace et
{

#if 0

Tensor boostFactor(const Tensor& average_activity, float target_activity, float boost_factor)
{
        return exp(target_activity - average_activity) * boost_factor; //TODO: Need to implement scalar-tensor operations to be complable
}

Tensor boostconst Tensor& activity, const Tensor& average_activity, float target_activity, float boost_factor)
{
	if(boost_factor == 0)
		return activity.copy();
	return cast(boostFactor(average_activity, target_activity, boost_factor)*activity, DType::Int32);	
}

#endif

}