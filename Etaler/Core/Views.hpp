#pragma once

#include "SmallVector.hpp"
#include "Shape.hpp"
#include "TensorImpl.hpp"

#include <variant>
#include <memory>

namespace et
{

struct Range
{
	Range() = default;
	Range(intmax_t start)
	{
		start_ = start;
		end_ = start+1;
	}

	Range(intmax_t start, intmax_t end)
	{
		start_ = start;
		end_ = end;

		if (start < 0) {
			start_from_back_ = true;
			end = -end;
		}

		if (end < 0) {
			end_from_back_ = true;
			end = -end;
		}
	}

	Range(intmax_t start, intmax_t end, bool start_from_back, bool end_from_back)
	{
		start_ = start;
		et_assert(end >= 0);
		end_ = end;
		start_from_back_ = start_from_back;
		end_from_back_ = end_from_back;
	}

	intmax_t start() const {return start_;}
	intmax_t end() const {return end_;}
	bool startFromBack() const {return start_from_back_;}
	bool endFromBack() const {return end_from_back_;}

protected:
	intmax_t start_ = 0;
	bool start_from_back_ = false;
	intmax_t end_ = 0;
	bool end_from_back_ = false;
	//intmax_t step_size_ = 1;
};

inline Range all()
{
	return Range(0, 0, false, true);
}

inline Range range(intmax_t start, intmax_t end)
{
	return Range(start, end);
}

inline Range range(intmax_t end)
{
	return Range(0, end);
}


}
