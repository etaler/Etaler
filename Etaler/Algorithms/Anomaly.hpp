#pragma once

#include <Etaler/Core/Tensor.hpp>

namespace et
{

inline float anomaly(const Tensor& pred, const Tensor& real)
{
	checkProperties(real.pimpl(), DType::Bool);
	checkProperties(pred.pimpl(), DType::Bool);
	et_check(real.shape() == pred.shape()
		, "The 1st and 2nd arguments should have to same shape");

	int should_predict = sum(real).item<int>();
	int not_predicted = sum((!pred) && real).item<int>();
	return float(not_predicted)/should_predict;
}

}