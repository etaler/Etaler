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

SpatialPoolerND::SpatialPoolerND(const Shape& input_shape, size_t kernel_size, size_t stride=1, float potential_pool_pct=0.75, size_t seed=42
	, float global_density = 0.15, float boost_factor = 0, Backend* b = defaultBackend())
{
	for(size_t i=0;i<input_shape.size();i++)
		et_assert(input_shape[i] >= (intmax_t)kernel_size);

	Shape output_shape(input_shape.size());
	for(size_t i=0;i<input_shape.size();i++)
		output_shape[i] = (input_shape[i]-kernel_size)/stride+1;
	output_shape_ = output_shape;

	size_t input_cell_num = input_shape.volume();
	size_t potential_pool_size = std::max((size_t)(std::pow(kernel_size, input_shape.size())*potential_pool_pct), size_t{1});

	pcg64 rng(seed);
	std::vector<int> all_input_cell = vector_range<int>(0, input_cell_num);
	Tensor indices = Tensor({input_shape}, all_input_cell.data(), b);
	all_input_cell.clear();

	connections_ = Tensor(output_shape+potential_pool_size, DType::Int32, b);
	permanences_ = Tensor(output_shape+potential_pool_size, DType::Float, b);

	for(size_t i=0;i<(size_t)output_shape.volume();i++) {
		svector<Range> write_loc;
		Shape loc = foldIndex(i, output_shape);
		for(size_t j=0;j<output_shape.size();j++)
			write_loc.push_back(loc[j]);

		svector<Range> read_loc(loc.size());
		for(size_t j=0;j<loc.size();j++) {
			intmax_t pos = loc[j]*stride;
			read_loc[j] = range(pos, pos+kernel_size);
		}

		std::vector<int> conns = indices.view(read_loc).toHost<int>();
		assert(conns.size() >= potential_pool_size);
		std::shuffle(conns.begin(), conns.end(), rng);
		std::sort(conns.begin(), conns.begin()+potential_pool_size);

		std::vector<float> permanences(potential_pool_size);
		std::normal_distribution<float> dist(connected_permanence_, 1);
		std::generate(permanences.begin(), permanences.end(), [&](){return std::max(std::min(dist(rng), 1.f), 0.f);});

		connections_.view(write_loc) = Tensor({(intmax_t)potential_pool_size}, conns.data());
		permanences_.view(write_loc) = Tensor({(intmax_t)potential_pool_size}, permanences.data());
	}
	average_activity_ = constant(output_shape, global_density);
}