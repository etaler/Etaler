#pragma once

#pragma once

#include <vector>
#include <set>

#include <Etaler/Core/Tensor.hpp>
#include <Etaler/Core/Backend.hpp>

namespace et
{

namespace encoder
{

static Tensor category(size_t category, size_t num_categories, size_t bits_per_category, Backend* backend=defaultBackend())
{
	et_check(category < num_categories, "Category " + std::to_string(category) + "is out of the encoder's range");

	std::vector<uint8_t> res(num_categories*bits_per_category);
	for(size_t i=0;i<bits_per_category;i++)
		res[i+category*bits_per_category] = 1;
	return Tensor({(intmax_t)(num_categories*bits_per_category)}, res.data(), backend);
}

}

namespace decoder
{

std::vector<size_t> category(const Tensor& t, size_t num_categories)
{
	requireProperties(t.pimpl(), DType::Bool);
	et_check(t.size()%num_categories == 0);

	std::vector<uint8_t> vec = t.toHost<uint8_t>();

	std::set<size_t> categories;
	size_t bits_per_category = vec.size()/num_categories;
	for(size_t i=0;i<vec.size();i++) {
		if(vec[i] != 0)
			categories.insert(i/bits_per_category);
	}

	std::vector<size_t> res;
	for(auto v : categories)
		res.push_back(v);
	return res;
}

}

}
