#include "SpatialPooler.hpp"
#include <Etaler/Core/Random.hpp>

using namespace et;

inline std::vector<size_t> vector_range(size_t start, size_t end)
{
	std::vector<size_t> v(end-start);
	for(size_t i=0;i<end-start;i++)
		v[i] = i;
	return v;
}


SpatialPooler::SpatialPooler(const Shape& input_shape, const Shape& output_shape, float potential_pool_pct, size_t seed
	, float global_density, Backend* b)
	: global_density_(global_density), input_shape_(input_shape), output_shape_(output_shape), backend_(b)
{
	if(potential_pool_pct > 1 or potential_pool_pct < 0)
		throw EtError("potential_pool_pct must be between 0~1, but get" + std::to_string(potential_pool_pct));

	size_t input_cell_num = input_shape.volume();
	size_t potential_pool_size = std::max((size_t)(input_cell_num*potential_pool_pct), size_t{1});

	//Initalize potential pool
	pcg64 rng(seed);
	std::vector<size_t> all_input_cell = vector_range(0, input_cell_num);

	std::vector<int32_t> connections(output_shape.volume()*potential_pool_size, -1);
	std::vector<float> permances(output_shape.volume()*potential_pool_size);

	auto clamp = [](float x){return std::min(1.f, std::max(x, 0.f));};
	for(intmax_t i=0;i<output_shape.volume();i++) {
		std::shuffle(all_input_cell.begin(), all_input_cell.end(), rng);
		std::sort(all_input_cell.begin(), all_input_cell.begin()+potential_pool_size);
		std::normal_distribution<float> dist(connected_permance_, 1);

		for(size_t j=0;j<potential_pool_size;j++) {
			connections[i*potential_pool_size+j] = all_input_cell[j];
			permances[i*potential_pool_size+j] = clamp(dist(rng));
		}
	}

	Shape s = output_shape + potential_pool_size;
	connections_ = backend_->createTensor(s, DType::Int32, connections.data());
	permances_ = backend_->createTensor(s, DType::Float, permances.data());
}

Tensor SpatialPooler::compute(const Tensor& x) const
{
	et_assert(x.shape() == input_shape_);

	Tensor t = backend_->overlapScore(x, connections_, permances_
		, connected_permance_, active_threshold_, false);

	Tensor res = backend_->globalInhibition(t, global_density_);

	return res;
}