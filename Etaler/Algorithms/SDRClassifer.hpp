#pragma once

#include <Etaler/Core/Tensor.hpp>

#include <vector>

namespace et
{

struct SDRClassifer
{
	SDRClassifer() = default;
	SDRClassifer(Shape input_shape, size_t num_classes)
		: input_shape_(input_shape), references_(num_classes, std::vector<uint32_t>(input_shape.volume()))
		, num_patterns_(num_classes)
	{
	}

	void addPattern(const Tensor& sdr, size_t class_id)
	{
		et_assert(sdr.shape() == input_shape_);
		et_assert(sdr.dtype() == DType::Bool);
		et_assert(class_id < references_.size());

		auto x = sdr.toHost<uint8_t>();
		for(size_t i=0;i<references_[class_id].size();i++)
			references_[class_id][i] += x[i];
		num_patterns_[class_id] ++;
	}

	size_t compute(const Tensor& x, float min_common_frac=0.75) const
	{
		size_t best_match_id = -1;
		size_t best_match = 0;
		for(size_t i=0;i<references_.size();i++) {
			size_t overlaps = 0;
			for(size_t j=0;j<references_[i].size();i++) {
				if(references_[i][j] >= min_common_frac*num_patterns_[i])
					overlaps++;
			}

			if(overlaps > best_match)
				std::tie(best_match_id, best_match) = std::pair(i, overlaps);
		}
		return best_match;
	}

	Shape input_shape_;
	std::vector<std::vector<uint32_t>> references_;
	std::vector<uint32_t> num_patterns_;
};

}