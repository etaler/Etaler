#pragma once

#include <arrayfire.h>

#include <Etaler/Core/Tensor.hpp>

namespace et
{

Tensor from_afarray(const af::array& arr, bool transpose=true)
{
	et_assert(arr.type() == f32 || arr.type() == s32 || arr.type() == b8);

        auto a = arr;
        if(transpose) //ArrayFire stores data in fortran order, we might want to transpose it to make it C order
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
                else
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
        else {
                auto ptr = a.host<uint8_t>(); //Some arrayfire quarks
                res = Tensor(s, ptr);
                af::freeHost(ptr);
        }

        return res;
}

af::array to_afarray(const Tensor& t, bool transpose=true)
{
	et_assert(t.dimentions() <= 4);
        af::dim4 dims;
        //Initalize the dims (not Initalized by default)
        for(int i=0;i<4;i++)
                dims[i] = 1;
        for(size_t i=0;i<t.dimentions();i++)
                dims[4-t.dimentions()+i] = t.shape()[i];

        af::dtype dtype = [](DType dtype) {
                if(dtype == DType::Float)
                        return f32;
                else if(dtype == DType::Int32)
                        return s32;
                else
                        return b8;
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
        else {
                auto v = t.toHost<uint8_t>();
                res.write(v.data(), v.size()*dtypeToSize(t.dtype()));
        }

        if(transpose)
                return res.T();
        return res;


}

}
