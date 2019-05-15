#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>

#include <Etaler/Core/Tensor.hpp>

namespace et
{

template <typename T>
Tensor from_xarray(const xt::xarray<T>& arr)
{
	auto s = arr.shape();
	return Tensor(Shape(s.begin(), s.end()), arr.data());
}

template <typename T>
xt::xarray<T> to_xarray(const Tensor& t)
{
	auto shape = t.shape();
	std::vector<size_t> s(shape.begin(), shape.end());
	auto vec = t.toHost<T>();
	return xt::adapt(vec.data(), s);
}

}