#pragma once

#include <Etaler/Core/Tensor.hpp>
#include <Etaler/Core/Serialize.hpp>

#include "Etaler_export.h"

#include <vector>
#include <mutex>

namespace et
{

struct ETALER_EXPORT SDRClassifer
{
	SDRClassifer() = default;
	SDRClassifer(Shape input_shape, size_t num_classes)
		: input_shape_(input_shape), reference_(zeros(Shape{(intmax_t)num_classes}+input_shape
			, DType::Int32))
		, num_patterns_(num_classes)
	{
	}

	size_t numCategories() const
	{
		return num_patterns_.size();
	}

	void addPattern(const Tensor& sdr, size_t class_id)
	{
		et_assert(sdr.shape() == input_shape_);
		et_assert(sdr.dtype() == DType::Bool);
		et_assert(class_id < num_patterns_.size());

		dirty_flag_ = true;
		reference_.view({(intmax_t)class_id}) = reference_.view({(intmax_t)class_id}) + sdr;
		num_patterns_[class_id]++;
	}

	size_t compute(const Tensor& x) const
	{
		const intmax_t num_classes = num_patterns_.size();

		// Grab a reference to ensure the current `mask_` isn't released in the case of race condition.
		// Then check if new patterns have been added. If so, recalculate the mask
		Tensor mask = mask_;
		if(dirty_flag_ == true) {
			mask_ = zeros_like(reference_.reshape({num_classes, input_shape_.volume()})).cast(DType::Bool);
			for(intmax_t i=0;i<num_classes;i++)
				mask_.view({i}) = globalInhibition(reference_.view({i}).flatten(), 1.f/num_classes);
			mask = mask_;
			dirty_flag_ = false;
		}

		assert(mask_.has_value());
		const auto overlap = logical_and(mask, x.reshape(Shape{1}+intmax_t(x.size())))
			.reshape({intmax_t(num_classes), input_shape_.volume()})
			.sum(1)
			.toHost<int>();

		et_assert(overlap.size() == size_t(num_classes));
		auto it = std::max_element(overlap.begin(), overlap.end());
		return std::distance(overlap.begin(), it);
	}

	StateDict states() const
	{
		return {{"input_shape", input_shape_}, {"reference", reference_}, {"num_patterns", num_patterns_}};
	}

	void loadState(const StateDict& states)
	{
		input_shape_ = std::any_cast<Shape>(states.at("input_shape"));
		reference_ = std::any_cast<Tensor>(states.at("reference"));
		num_patterns_ = std::any_cast<std::vector<int>>(states.at("num_patterns"));
	}

	SDRClassifer to(Backend* b) const
	{
		SDRClassifer c = *this;
		assert(c.reference_.size() == reference_.size());
		c.reference_ = reference_.to(b);
		return c;
	}

	SDRClassifer copy() const
	{
		if(reference_.size() == 0)
			return *this;
		return to(reference_.backend());
	}

	Shape input_shape_;
	Tensor reference_;
	std::vector<int> num_patterns_;

private:
	// mask_ and dirty_flag_ are variables for caching expensive-to-compute values
	// that is used in the inference process
	// NOTE: They *should* be thrad safe. (But the internal functions are parallel anyway)
	mutable Tensor mask_;
	mutable bool dirty_flag_ = true;
};

// SDRClassifer in Etaler is CLAClassifer in NuPIC
// SDRClassifer from NuPIC is not implemented
using CLAClassifer = SDRClassifer;

}
