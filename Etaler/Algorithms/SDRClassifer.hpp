#pragma once

#include <Etaler/Core/Tensor.hpp>

#include <vector>

namespace et
{

struct SDRClassifer
{
	SDRClassifer() = default;
	SDRClassifer(Shape input_shape, size_t num_classes)
		: input_shape_(input_shape), references_(num_classes, zeros(input_shape))
		, num_patterns_(num_classes)
	{
	}

	void addPattern(const Tensor& sdr, size_t class_id)
	{
		et_assert(sdr.shape() == input_shape_);
		et_assert(sdr.dtype() == DType::Bool);
		et_assert(class_id < references_.size());

		references_[class_id] = references_[class_id] + sdr;
	}

	size_t compute(const Tensor& x, float min_common_frac=0.75) const
	{
		size_t best_match_id = -1;
		size_t best_match = 0;
		for(size_t i=0;i<references_.size();i++) {
			int threshold = num_patterns_[i]*min_common_frac;
			size_t overlaps = sum((references_[i] > threshold) && x).toHost<int32_t>()[0];

			if(overlaps > best_match)
				std::tie(best_match_id, best_match) = std::pair(i, overlaps);
		}
		return best_match;
	}

	StateDict states() const
	{
		return {{"input_shape", input_shape_}, {"references", references_}, {"num_patterns", num_patterns_}};
	}

	void loadState(const StateDict& states)
	{
		input_shape_ = std::any_cast<Shape>(states["input_shape"]);
		references_ = std::any_cast<Tensor>(states["references"]);
		num_patterns_ = std::any_cast<std::vector<int>>(states["num_patterns"]);
	}

	SDRClassifer to(Backend* b)
	{
		SDRClassifer c = *this;
		c.references_ = references_.to(b);
	}



	Shape input_shape_;
	std::vector<Tensor> references_;
	std::vector<int> num_patterns_;
};

}