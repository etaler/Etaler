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
	std::tie(connections_, permanences_) = F::gusianRandomSynapse(input_shape, output_shape, potential_pool_pct
		, 0.11, 1, seed, b);
	average_activity_ = constant(output_shape, global_density);
}

Tensor SpatialPooler::compute(const Tensor& x) const
{
	et_check(x.shape() == input_shape_, "Input tensor shape " + to_string(x.shape()) +" does not match expected shape " + to_string(input_shape_));

	Tensor activity = cellActivity(x, connections_, permanences_
		, connected_permanence_, active_threshold_, false);

	if(boost_factor_ != 0)
		activity = boost(activity, average_activity_, global_density_, boost_factor_);

	Tensor res = globalInhibition(activity, global_density_);
	assert(output_shape_ == res.shape());

	return res;
}

void SpatialPooler::learn(const Tensor& x, const Tensor& y)
{
	et_assert(x.shape() == input_shape_);

	learnCorrilation(x, y, connections_, permanences_, permanence_inc_, permanence_dec_);

	if(boost_factor_ != 0)
		average_activity_ = average_activity_*0.9f + y * 0.1f;
}

void SpatialPooler::loadState(const StateDict& states)
{
	permanence_inc_ = std::any_cast<float>(states.at("permanence_inc"));
	permanence_dec_ = std::any_cast<float>(states.at("permanence_dec"));
	connected_permanence_ = std::any_cast<float>(states.at("connected_permanence"));
	active_threshold_ = std::any_cast<int>(states.at("active_threshold"));
	global_density_ = std::any_cast<float>(states.at("global_density"));
	input_shape_ = std::any_cast<Shape>(states.at("input_shape"));
	output_shape_ = std::any_cast<Shape>(states.at("output_shape"));
	connections_ = std::any_cast<Tensor>(states.at("connections"));
	permanences_ = std::any_cast<Tensor>(states.at("permanences"));
	average_activity_ = std::any_cast<Tensor>(states.at("average_activity"));
	boost_factor_ = std::any_cast<float>(states.at("boost_factor"));
}

SpatialPooler SpatialPooler::to(Backend* b) const
{
	SpatialPooler sp = *this;
	sp.connections_ = connections_.to(b);
	sp.permanences_ = permanences_.to(b);
	sp.average_activity_ = average_activity_.to(b);

	return sp;
}
