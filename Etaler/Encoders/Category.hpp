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
	et_assert(category < num_categories);

	std::vector<uint8_t> res(num_categories*bits_per_category);
	for(size_t i=0;i<bits_per_category;i++)
		res[i+category*bits_per_category] = 1;
	return backend->createTensor({(intmax_t)(num_categories*bits_per_category)}, DType::Bool, res.data());
}

}

namespace decoder
{

std::vector<size_t> category(const Tensor& t, size_t num_categories)
{
	et_assert(t.size()%num_categories == 0);
	et_assert(t.dtype() == DType::Bool);

	std::vector<uint8_t> vec(t.size());
	t.backend()->copyToHost(t.pimpl(), vec.data());

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
