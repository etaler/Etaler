#pragma once

#include <Etaler/Core/Tensor.hpp>

#include <numeric>

namespace et
{

float anomaly(const Tensor& pred, const Tensor& real)
{
	et_assert(real.dtype() == DType::Bool);
	et_assert(pred.dtype() == DType::Bool);
	et_assert(real.shape() == pred.shape());

	auto pred_bits = pred.toHost<uint8_t>();
	auto real_bits = real.toHost<uint8_t>();

	size_t not_predicted = 0;
	size_t should_predict = 0;
	for(size_t i=0;i<pred_bits.size();i++) {
		not_predicted += (pred_bits[i] == 0 && real_bits[i] == 1);
		should_predict += real_bits[i];
	}

	return ((float)not_predicted)/should_predict;
}

}