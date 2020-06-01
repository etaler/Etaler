#pragma once

#include <Etaler/Core/Tensor.hpp>

namespace et
{

static float anomaly(const Tensor& pred, const Tensor& real)
{
	et_assert(real.dtype() == DType::Bool);
	et_assert(pred.dtype() == DType::Bool);
	et_assert(real.shape() == pred.shape());

	int should_predict = sum(real).item<int>();
	int not_predicted = sum((!pred) && real).item<int>();
	return float(not_predicted)/should_predict;
}

}