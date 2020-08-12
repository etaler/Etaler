#pragma once

#include <arrayfire.h>

#include <Etaler/Core/Tensor.hpp>

namespace et
{

Tensor from_afarray(const af::array& arr, bool transpose=true)
{
	et_check(arr.type() == f32 || arr.type() == s32 || arr.type() == b8
		, "Etaler does not support the data type");

	auto a = arr;
	//ArrayFire stores data in fortran order, we might want to transpose it to make it C order
	//But also AF transposes 1D arrays into 2D matrices.
	if(transpose && arr.dims().ndims() != 1)
		a = af::transpose(arr);

	//Convert to et::Shape
	Shape s;
	auto dims = a.dims();
	for(dim_t i=0;i<dims.ndims();i++)
		s.push_back(dims[i]);

	DType dtype = [](auto type) {
		if(type == f32)
			return DType::Float;
		else if(type == s32)
			return DType::Int32;
		else if(type == b8)
			return DType::Bool;
	}(arr.type());

	//Copy data from AF to Etaler
	Tensor res;
	if(dtype == DType::Float) {
		auto ptr = a.host<float>();
		res = Tensor(s, ptr);
		af::freeHost(ptr);
	}
	else if(dtype == DType::Int32) {
		auto ptr = a.host<int>();
		res = Tensor(s, ptr);
		af::freeHost(ptr);
	}
	else if(dtype == DType::Bool) {
		auto ptr = a.host<uint8_t>(); //Some arrayfire quarks
		res = Tensor(s, ptr);
		af::freeHost(ptr);
	}
	else
		throw EtError("from_afarray failed. Data type contained in ArrayFire array is not supported by Etaler.");

	return res;
}

af::array to_afarray(const Tensor& t, bool transpose=true)
{
	et_assert(t.dimensions() <= 4);
	af::dim4 dims;
	//Initalize the dims (not Initalized by default)
	for(int i=0;i<4;i++)
		dims[i] = 1;
	for(size_t i=0;i<t.dimensions();i++)
		dims[t.dimensions()-i] = t.shape()[i];

	af::dtype dtype = [](DType dtype) {
		if(dtype == DType::Float)
			return f32;
		else if(dtype == DType::Int32)
			return s32;
		else if(dtype == DType::Bool)
			return b8;
		else
			throw EtError("to_afarray failed. Data type held by et::Tensor is not supported by ArrayFire.");
	}(t.dtype());
	af::array res(dims, dtype);

	if(dtype == f32) {
		auto v = t.toHost<float>();
		res.write(v.data(), v.size()*dtypeToSize(t.dtype()));
	}
	else if(dtype == s32) {
		auto v = t.toHost<int32_t>();
		res.write(v.data(), v.size()*dtypeToSize(t.dtype()));
	}
	else if(dtype == b8) {
		auto v = t.toHost<uint8_t>();
		res.write(v.data(), v.size()*dtypeToSize(t.dtype()));
	}
	else
		ASSERT(false && "to_afarray: should not reach here.");

	if(transpose)
		return res.T();
	return res;


}

}
