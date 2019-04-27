#include "CPUBackend.hpp"

#include <numeric>

#include <tbb/tbb.h>

using namespace et;


//Helper functions
template <typename T>
void apply_permutation_in_place(
	T* start,
	T* end,
	const std::vector<std::size_t>& p)
{
	size_t size = std::distance(start, end);
	et_assert(size == p.size());
	std::vector<bool> done(size);
	for (size_t i=0; i < size; ++i) {
		if (done[i])
			continue;
		done[i] = true;
		size_t prev_j = i;
		size_t j = p[i];
		while (i != j) {
			std::swap(start[prev_j], start[j]);
			done[j] = true;
			prev_j = j;
			j = p[j];
		}
	}
}


const void* CPUTensor::data() const
{
	assert(dtype() != DType::Unknown);
	return std::visit([](const auto& v){return (const void*)v.data();}, storage_);
}

void CPUBackend::overlapScore(const TensorImpl* x, const TensorImpl* connections, const TensorImpl* permeances
	, float connected_permeance, size_t active_threshold, TensorImpl* y, bool has_unconnected_synapse)
{
	//Checks the input are sane
	et_assert(points_to<CPUTensor>(x));
	et_assert(points_to<CPUTensor>(connections));
	et_assert(points_to<CPUTensor>(permeances));
	et_assert(points_to<CPUTensor>(y));

	et_assert(x->dtype() == DType::Bool);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);
	et_assert(y->dtype() == DType::Int32);
	et_assert(connections->shape() == permeances->shape());


	const bool* input = (const bool*)x->data();
	const int32_t* synapses = (const int32_t*)connections->data();
	const float* synapse_strengths = (float*)permeances->data();
	int32_t* result = (int32_t*)y->data();

	size_t max_connections_per_cell = connections->shape().back();
	size_t num_cells = connections->size()/max_connections_per_cell;

	tbb::parallel_for(size_t(0), num_cells, [&](size_t i) {
		size_t sum = 0;
		for(size_t j=0;j<max_connections_per_cell;j++) {
			size_t index = i*max_connections_per_cell+j;
			size_t target = synapses[index];
			float strength = synapse_strengths[index];

			assert(target < x->size());

			if(target == (size_t)-1)
				break;
			if(input[target] == 1 && strength > connected_permeance)
				sum += 1;
		}
		if(sum >= active_threshold)
			result[i] = sum;
		else
			result[i] = 0;
	});
}

void CPUBackend::learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections, TensorImpl* permeances, float perm_inc, float perm_dec)
{
	et_assert(points_to<CPUTensor>(x));
	et_assert(points_to<CPUTensor>(connections));
	et_assert(points_to<CPUTensor>(permeances));
	et_assert(points_to<CPUTensor>(learn));

	et_assert(connections->shape() == permeances->shape());
	et_assert(x->shape() == learn->shape());
	et_assert(x->dtype() == DType::Bool);
	et_assert(learn->dtype() == DType::Bool);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);

	const bool* input = (const bool*)x->data();
	const bool* learning = (const bool*)learn->data();
	const int32_t* synapses = (const int32_t*)connections->data();
	float* synapse_strengths = (float*)permeances->data();

	size_t max_connections_per_cell = connections->shape().back();
	size_t num_cells = connections->size()/max_connections_per_cell;

	tbb::parallel_for(size_t(0), connections->size(), [&](size_t i) {
		if(learning[i] == false)
			return;

		for(size_t j=0; j<max_connections_per_cell;j++) {
			size_t idx = i*max_connections_per_cell+j;
			auto connection = synapses[idx];
			if(connection == -1)
				break;
			ASSERT((size_t)connection < num_cells);

			float& perm = synapse_strengths[idx];
			if(input[connection] == true)
				perm += perm_inc;
			else
				perm -= perm_dec;

			perm = std::max(std::min(perm, 1.f), 0.f);
		}
	});
}

void CPUBackend::globalInhibition(const TensorImpl* x, TensorImpl* y, float fraction)
{
	et_assert(points_to<CPUTensor>(x));
	et_assert(points_to<CPUTensor>(y));

	et_assert(x->dtype() == DType::Int32);
	et_assert(y->dtype() == DType::Bool);
	et_assert(x->shape() == y->shape());

	const int32_t* input = (const int32_t*)x->data();
	bool* output = (bool*)y->data();

	std::vector<std::pair<int32_t, size_t>> v;
	size_t target_size = x->size()*fraction;
	v.reserve(target_size);//Some sane value
	for(size_t i=0;i<x->size();i++) {
		if(input[i] != 0)
			v.push_back({input[i], i});
	}
	std::sort(v.begin(), v.end(), [](const auto& a, const auto&b){return a.first > b.first;});

	for(size_t i=0;i<y->size();i++)
		output[i] = false;
	int32_t min_accept_val = v[std::min((target_size==0? 0 : target_size-1), v.size()-1)].first;
	auto bound_end = std::upper_bound(v.begin(), v.end(), min_accept_val, [](const auto& a, const auto& b){return a > b.first;});

	for(auto it=v.begin();it!=bound_end;++it)
		output[it->second] = true;
}

template <typename Ret, typename Op>
static Ret run(const CPUTensor& t, Op op)
{
	if(t.dtype() == DType::Bool)
		return op((const bool*)t.data());
	else if(t.dtype() == DType::Int32)
		return op((int32_t*)t.data());
	else if(t.dtype() == DType::Float)
		return op((float*)t.data());
	else
		throw EtError("Cannot cast");
}

template <typename To, typename From>
static std::vector<To> castData(const From* ptr, size_t n)
{
	return std::vector<To>(ptr, ptr+n);
}

std::shared_ptr<TensorImpl> CPUBackend::cast(const TensorImpl* x, DType toType)
{
	et_assert(points_to<CPUTensor>(x));
	const CPUTensor* p = dynamic_cast<const CPUTensor*>(x);
	const CPUTensor& t = *p;
	return run<std::shared_ptr<TensorImpl>>(t, [&x, toType, this](const auto* ptr){
		if(toType == DType::Bool) {
			auto castedData = castData<uint8_t>(ptr, x->size());
			return createTensor(x->shape(), toType, castedData.data());
		}
		else if(toType == DType::Int32) {
			auto castedData = castData<int32_t>(ptr, x->size());
			return createTensor(x->shape(), toType, castedData.data());
		}
		else if(toType == DType::Float){
			auto castedData = castData<float>(ptr, x->size());
			return createTensor(x->shape(), toType, castedData.data());
		}
		else
			throw EtError("Cannot cast");
	});
}

void CPUBackend::copyToHost(const TensorImpl* pimpl, void* ptr)
{
	et_assert(points_to<CPUTensor>(pimpl));
	const CPUTensor* t = dynamic_cast<const CPUTensor*>(pimpl);
	if(t == nullptr)
		throw EtError("Cannot copy to host memory: Tensor/Backend mismach");
	memcpy(ptr, t->data(), t->size()*dtypeToSize(t->dtype()));
}

std::shared_ptr<TensorImpl> CPUBackend::copy(const TensorImpl* x)
{
	et_assert(points_to<CPUTensor>(x));
	return createTensor(x->shape(), x->dtype(), reinterpret_cast<const CPUTensor*>(x)->data());
}

void CPUBackend::sortSynapse(TensorImpl* connections, TensorImpl* permeances)
{
	et_assert(connections->shape() == permeances->shape());
	et_assert(points_to<CPUTensor>(connections));
	et_assert(points_to<CPUTensor>(permeances));
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);

	size_t max_synapse_per_cell = connections->shape().back();
	size_t num_cells = connections->size()/max_synapse_per_cell;

	uint32_t* conns = (uint32_t*)connections->data(); //HACK: -1s should be at the end of the arrays.
	float* perms = (float*)permeances->data();

	tbb::parallel_for(size_t(0), num_cells, [&](size_t i) {
		size_t start_index = i*max_synapse_per_cell;
		size_t end_index = (i+1)*max_synapse_per_cell;

		std::vector<size_t> sort_indices(max_synapse_per_cell);
		std::iota(sort_indices.begin(), sort_indices.end(), 0);
		std::sort(sort_indices.begin(), sort_indices.end(),
			[&](size_t i, size_t j)->bool {
				return conns[i+start_index] < conns[j+start_index];
			});
		apply_permutation_in_place(conns+start_index, conns+end_index, sort_indices);
		apply_permutation_in_place(perms+start_index, perms+end_index, sort_indices);
	});
}

void CPUBackend::applyBurst(const TensorImpl* x, TensorImpl* y)
{
	et_assert(points_to<CPUTensor>(x));
	et_assert(points_to<CPUTensor>(y));
	et_assert(x->dtype() == DType::Bool);
	et_assert(y->dtype() == DType::Bool);

	{Shape s = y->shape(); s.pop_back();et_assert(x->shape() == s);}

	const bool* in = (const bool*)x->data();
	bool* out = (bool*)y->data();

	size_t column_size = y->shape().back();
	tbb::parallel_for(size_t(0), x->size(), [&](size_t i){
		if(in[i] == false)
			return;
		if(std::accumulate(out+i*column_size, out+(i+1)*column_size, 0) == 0)
			std::generate(out+i*column_size, out+(i+1)*column_size, [](){return 1;});
	});
}
