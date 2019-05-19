#pragma once

#include <Etaler/Core/Tensor.hpp>

namespace et
{

static float anomaly(const Tensor& pred, const Tensor& real)
{
	et_assert(real.dtype() == DType::Bool);
	et_assert(pred.dtype() == DType::Bool);
	et_assert(real.shape() == pred.shape());

	Tensor should_predict = sum(real);
	Tensor not_predicted = sum(!pred && real).cast(DType::Float);
	return (not_predicted/should_predict).toHost<float>()[0];
}

}