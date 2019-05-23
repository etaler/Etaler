#include "SpatialPooler.hpp"
#include <Etaler/Core/Random.hpp>
#include "Boost.hpp"

using namespace et;

template <typename T=size_t>
inline std::vector<T> vector_range(size_t start, size_t end)
{
	std::vector<T> v(end-start);
	for(size_t i=0;i<end-start;i++)
		v[i] = i;
	return v;
}


SpatialPooler::SpatialPooler(const Shape& input_shape, const Shape& output_shape, float potential_pool_pct, size_t seed
	, float global_density, float boost_factor, Backend* b)
	: global_density_(global_density), boost_factor_(boost_factor), input_shape_(input_shape), output_shape_(output_shape)
{
	if(potential_pool_pct > 1 || potential_pool_pct < 0)
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

	if(boost_factor != 0)
		average_activity_ = constant(input_shape, global_density);

	Shape s = output_shape + potential_pool_size;
	connections_ = Tensor(s, connections.data(), b);
	permances_ = Tensor(s, permances.data(), b);
}

SpatialPooler::SpatialPooler(const Shape& input_shape, size_t kernel_size, size_t stride, float potential_pool_pct, size_t seed
	, float global_density, float boost_factor, Backend* b)
	:global_density_(global_density), boost_factor_(boost_factor), input_shape_(input_shape)
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
	permances_ = Tensor(output_shape+potential_pool_size, DType::Float, b);

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
		
		std::vector<float> permances(potential_pool_size);
		std::normal_distribution<float> dist(connected_permance_, 1);
		std::generate(permances.begin(), permances.end(), [&](){return std::max(std::min(dist(rng), 1.f), 0.f);});

		connections_.view(write_loc) = Tensor({(intmax_t)potential_pool_size}, conns.data());
		permances_.view(write_loc) = Tensor({(intmax_t)potential_pool_size}, permances.data());
	}
}

Tensor SpatialPooler::compute(const Tensor& x) const
{
	et_assert(x.shape() == input_shape_);

	Tensor activity = cellActivity(x, connections_, permances_
		, connected_permance_, active_threshold_, false);

	if(boost_factor_ != 0)
		activity = boost(activity, average_activity_, global_density_, boost_factor_);

	Tensor res = globalInhibition(activity, global_density_);

	return res;
}

void SpatialPooler::learn(const Tensor& x, const Tensor& y)
{
	et_assert(x.shape() == input_shape_);
	et_assert(y.shape() == input_shape_);
	learnCorrilation(x, y, connections_, permances_, permance_inc_, permance_dec_);

	if(boost_factor_ != 0)
		average_activity_ = average_activity_*0.9f + y * 0.1f;
}

void SpatialPooler::loadState(const StateDict& states)
{
	permance_inc_ = std::any_cast<float>(states.at("permance_inc"));
	permance_dec_ = std::any_cast<float>(states.at("permance_dec"));
	connected_permance_ = std::any_cast<float>(states.at("connected_permance"));
	active_threshold_ = std::any_cast<int>(states.at("active_threshold"));
	global_density_ = std::any_cast<float>(states.at("global_density"));
	input_shape_ = std::any_cast<Shape>(states.at("input_shape"));
	output_shape_ = std::any_cast<Shape>(states.at("output_shape"));
	connections_ = std::any_cast<Tensor>(states.at("connections"));
	permances_ = std::any_cast<Tensor>(states.at("permances"));
	average_activity_ = std::any_cast<Tensor>(states.at("average_activity"));
	boost_factor_ = std::any_cast<float>(states.at("boost_factor"));
}

SpatialPooler SpatialPooler::to(Backend* b) const
{
	SpatialPooler sp = *this;
	sp.connections_ = connections_.to(b);
	sp.permances_ = permances_.to(b);
	sp.average_activity_ = average_activity_.to(b);

	return sp;
}
