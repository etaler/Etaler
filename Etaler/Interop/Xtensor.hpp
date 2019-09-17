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
	//Handle the case of bool
	using DataType = typename std::conditional<std::is_same_v<T, bool>, uint8_t, T>::type;
	auto vec = t.toHost<DataType>();
	return xt::adapt((const T*)vec.data(), s);
}

}
