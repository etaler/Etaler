#include "SpatialPoolerND.hpp"
#include <Etaler/Core/Random.hpp>

using namespace et;

template <typename T=size_t>
inline std::vector<T> vector_range(size_t start, size_t end)
{
	std::vector<T> v(end-start);
	for(size_t i=0;i<end-start;i++)
		v[i] = i;
	return v;
}

SpatialPoolerND::SpatialPoolerND(const Shape& input_shape, size_t kernel_size, size_t stride, float potential_pool_pct, size_t seed
	, float global_density, float boost_factor, Backend* b)
{
	for(size_t i=0;i<input_shape.size();i++)
		et_check(input_shape[i] >= (intmax_t)kernel_size, "Input dimension" + std::to_string(i) + " is smaller than the kernel size");

	size_t ndims = input_shape.size();
	Shape output_shape = convResultShape(input_shape, Shape(ndims, intmax_t(kernel_size)), Shape(ndims, intmax_t(stride)));

	std::tie(connections_, permanences_) = F::gusianRandomSynapseND(input_shape, kernel_size, stride, potential_pool_pct, 0.11
		, 1, seed, b);
	average_activity_ = constant(output_shape, global_density);

	//HACK: Set SP core parameters because derrived calss cannot set them in initalization list
	global_density_ = global_density;
	boost_factor_ = boost_factor_;
	input_shape_ = input_shape;
	output_shape_ = output_shape;
}