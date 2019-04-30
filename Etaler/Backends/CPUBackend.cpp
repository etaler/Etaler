#include "CPUBackend.hpp"
#include "Etaler/Core/Views.hpp"
#include "Etaler/Core/Random.hpp"

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

template <typename T>
void apply_permutation(
	const T* start,
	const T* end,
	T* dest,
	const std::vector<std::size_t>& p)
{
	std::vector<T> sorted_vec(p.size());
	std::transform(p.begin(), p.end(), dest,
		[&](std::size_t i){ return start[i]; });
}


const void* CPUTensor::data() const
{
	assert(dtype() != DType::Unknown);
	return std::visit([](const auto& v){return (const void*)v;}, storage_);
}

std::shared_ptr<TensorImpl> CPUBackend::overlapScore(const TensorImpl* x, const TensorImpl* connections, const TensorImpl* permeances
	, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse)
{
	//Checks the input are sane
	et_assert(points_to<CPUTensor>(x));
	et_assert(points_to<CPUTensor>(connections));
	et_assert(points_to<CPUTensor>(permeances));

	et_assert(x->dtype() == DType::Bool);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);
	et_assert(connections->shape() == permeances->shape());
	et_assert(connections->dimentions() >= 2);

	Shape s = connections->shape();
	s.pop_back();
	auto y = createTensor(s, DType::Int32);


	const bool* input = (const bool*)x->data();
	const int32_t* synapses = (const int32_t*)connections->data();
	const float* synapse_strengths = (float*)permeances->data();
	int32_t* result = (int32_t*)y->data();

	size_t max_connections_per_cell = connections->shape().back();
	size_t num_cells = connections->size()/max_connections_per_cell;


	size_t block_size = std::min(size_t(16), (size_t)num_cells);
	tbb::parallel_for(tbb::blocked_range<size_t>(size_t(0), num_cells, block_size), [&](const auto& r) {
		for(size_t i=r.begin();i!=r.end();i++) {
			size_t sum = 0;
			for(size_t j=0;j<max_connections_per_cell;j++) {
				size_t index = i*max_connections_per_cell+j;
				int32_t target = synapses[index];

				if(target == -1)
					break;

				assert(target < (int32_t)x->size());

				if(input[target] == false)
					continue;

				float strength = synapse_strengths[index];
				if(strength > connected_permeance)
					sum += 1;
			}
			if(sum >= active_threshold)
				result[i] = sum;
			else
				result[i] = 0;
		}
	});

	return y;
}

void CPUBackend::learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections, TensorImpl* permeances
	, float perm_inc, float perm_dec, bool has_unconnected_synapse)
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

	tbb::parallel_for(size_t(0), learn->size(), [&](size_t i) {
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

std::shared_ptr<TensorImpl> CPUBackend::globalInhibition(const TensorImpl* x, float fraction)
{
	et_assert(points_to<CPUTensor>(x));

	et_assert(x->dtype() == DType::Int32);

	auto y = createTensor(x->shape(), DType::Bool);

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
	return y;
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
			std::transform(castedData.begin(), castedData.end(), castedData.begin(), [](auto v) {return bool(v);});
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

std::shared_ptr<TensorImpl> CPUBackend::applyBurst(const TensorImpl* x, const TensorImpl* s)
{
	et_assert(points_to<const CPUTensor>(x));
	et_assert(points_to<const CPUTensor>(s));
	et_assert(x->dtype() == DType::Bool);
	et_assert(s->dtype() == DType::Bool);

	Shape shape = s->shape();
	shape.pop_back();
	et_assert(shape == x->shape());

	auto y = createTensor(s->shape(), DType::Bool);

	const bool* in = (const bool*)x->data();
	const bool* state = (const bool*)s->data();
	bool* out = (bool*)y->data();

	size_t column_size = y->shape().back();
	tbb::parallel_for(size_t(0), x->size(), [&](size_t i) {
		if(in[i] == false)
			std::generate(out+i*column_size, out+(i+1)*column_size, [](){return 0;});
		else
		{
			if(std::accumulate(state+i*column_size, state+(i+1)*column_size, 0) == 0)
				std::generate(out+i*column_size, out+(i+1)*column_size, [](){return 1;});
			else
				std::copy(state+i*column_size, state+(i+1)*column_size, out+i*column_size);
		}
	});
	return y;
}

std::shared_ptr<TensorImpl> CPUBackend::reverseBurst(const TensorImpl* x)
{
	et_assert(points_to<const CPUTensor>(x));
	et_assert(x->dtype() == DType::Bool);

	size_t cells_per_column = x->shape().back();
	size_t num_columns = x->size()/cells_per_column;
	static pcg64 rng; //Static so the behavor hangees every time, breaking symmetry
	std::uniform_int_distribution<size_t> dist(0, cells_per_column-1);

	auto y = createTensor(x->shape(), DType::Bool);

	const bool* in = (const bool*) x->data();
	bool* out = (bool*) y->data();

	tbb::parallel_for(size_t(0), num_columns, [&](size_t i) {
		if(std::accumulate(in+i*cells_per_column, in+(i+1)*cells_per_column, size_t(0)) == cells_per_column) {
			std::generate(out+i*cells_per_column, out+(i+1)*cells_per_column, [](){return 0;});
			out[i*cells_per_column+dist(rng)] = 1;
		}
		else
			std::copy(in+i*cells_per_column, in+(i+1)*cells_per_column, out+i*cells_per_column);
	});
	return y;

}

void CPUBackend::growSynapses(const TensorImpl* x, const TensorImpl* y, TensorImpl* connections
	, TensorImpl* permeances, float initial_perm)
{
	et_assert(points_to<const CPUTensor>(x));
	et_assert(points_to<const CPUTensor>(y));
	et_assert(points_to<CPUTensor>(connections));
	et_assert(points_to<CPUTensor>(permeances));

	et_assert(x->dtype() == DType::Bool);
	et_assert(y->dtype() == DType::Bool);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);

	et_assert(x->shape() == y->shape());
	et_assert(connections->shape() == permeances->shape());
	Shape s = connections->shape();
	s.pop_back();
	et_assert(s == y->shape());

	size_t max_synapses_per_cell = connections->shape().back();
	size_t input_cell_count = x->size();

	const bool* in = (const bool*) x->data();
	const bool* out = (const bool*) y->data();
	int32_t* conns = (int32_t*)connections->data();
	float* perms = (float*)permeances->data();

	std::vector<int> on_bits;
	on_bits.reserve(input_cell_count*0.1);
	for(size_t i=0;i<input_cell_count;i++) {
		if(in[i] == true)
			on_bits.push_back(i);
	}

	size_t block_size = std::min(size_t(16), (size_t)y->shape().back());
	tbb::parallel_for(tbb::blocked_range<size_t>(size_t(0), y->size(), block_size), [&](const auto& r) {
		for(size_t i=r.begin();i!=r.end();i++) {
			if(out[i] == 0)
				return;

			int32_t* synapses = conns+i*max_synapses_per_cell;
			float* strengths = perms+i*max_synapses_per_cell;
			int32_t* end = synapses+max_synapses_per_cell;

			if(synapses[max_synapses_per_cell-1] != -1) //If there is no space for new synapse. Ignore
				return;

			int32_t* it = std::lower_bound(synapses, end, -1);
			size_t avliable_space = end - it;
			size_t used_space = it - synapses;

			size_t write_idx = it - synapses;
			size_t read_idx = 0;

			for(size_t j=0;avliable_space!=0 && j < on_bits.size();j++) {
				bool connected = false;
				for(;read_idx<used_space;read_idx++) {
					if(synapses[read_idx] == on_bits[j]) {
						connected = true;
						break;
					}
					if(synapses[read_idx] > on_bits[j])
						break;
				}

				if(connected == false) {
					synapses[write_idx] = on_bits[j];
					strengths[write_idx] = initial_perm;
					avliable_space--;
					write_idx++;
				}
			}

			std::vector<size_t> sort_indices(write_idx);
			std::iota(sort_indices.begin(), sort_indices.begin()+write_idx, 0);
			std::sort(sort_indices.begin(), sort_indices.begin()+write_idx,
				[&](size_t i, size_t j)->bool {
					return ((uint32_t*)synapses)[i] < ((uint32_t*)synapses)[j];
				});
			apply_permutation_in_place(synapses, synapses+write_idx, sort_indices);
			apply_permutation_in_place(strengths, strengths+write_idx, sort_indices);
		}
	});
}

template <typename Op>
void ndIter(const Shape& s, Op op, Shape current=Shape())
{
	if(current.size() == s.size())
		op(current);
	else {
		size_t idx = current.size();
		for(intmax_t i=0;i<s[idx];i++) {
			Shape next = current + i;
			ndIter(s, op, next);
		}
	}
}

template <typename IdxType, typename ShapeType>
inline size_t unfoldIndex(const IdxType& index, const ShapeType& shape)
{
	size_t s = 0;
	size_t v = 1;
	assert(index.size() == shape.size());
	for(int i=(int)index.size()-1;i>=0;i--) {
		v *= (i==(int)index.size()-1?1:shape[i+1]);
		s += index[i] * v;
	}

	return s;
}

template <typename ShapeType>
Shape foldIndex(size_t index, const ShapeType& shape)
{
	assert(shape.size() != 0);
	Shape v;
	v.resize(shape.size());
	size_t acc = 1;
	for(int i=(int)shape.size()-1;i>=0;i--) {
		acc *= shape[i];
		v[i] = acc;
	}
	Shape res;
	res.resize(v.size());
	for(size_t i=1;i<v.size();i++) {
		res[i-1] = index/v[i];
		index = index%v[i];
	}
	res.back() = index;
	return res;
}

template <typename T>
T getValue(const Shape& s, const TensorImpl* t)
{
	if(points_to<const CPUTensor>(t) == true) {
		const T* ptr = (const T*) t->data();
		return ptr[unfoldIndex(s, t->shape())];
	}

	et_assert(points_to<const ViewTensor>(t) == true);
	T value;
	std::visit([&](const auto& view) {
		Shape parent_loc;
		const TensorImpl* parent = reinterpret_cast<const ViewTensor*>(t)->parent_.get();
		using ViewType = std::decay_t<decltype(view)>;
		if constexpr(std::is_same_v<ViewType, ReshapeView>)
			parent_loc = foldIndex(unfoldIndex(s, t->shape()), parent->shape());
		else
			throw EtError("Not supported");

		value = getValue<T>(parent_loc, parent);

	}, reinterpret_cast<const ViewTensor*>(t)->view_);

	return value;

}

std::shared_ptr<TensorImpl> CPUBackend::realize(const TensorImpl* x)
{
	if(points_to<const CPUTensor>(x) == true)
		return copy(x);
	if(points_to<const ViewTensor>(x) == false)
		throw EtError("Cannot realize tensor, not a CPUTensor or ViewTensor");
	auto res = createTensor(x->shape(), x->dtype());
	ndIter(x->shape(), [&](const Shape& idx) {
		if(x->dtype() == DType::Int32) {
			auto ptr = (int32_t*)res->data();
			ptr[unfoldIndex(idx, res->shape())] = getValue<int32_t>(idx, x);
		}
		else if(x->dtype() == DType::Float) {
			auto ptr = (float*)res->data();
			ptr[unfoldIndex(idx, res->shape())] = getValue<float>(idx, x);
		}
		else if(x->dtype() == DType::Bool) {
			auto ptr = (uint8_t*)res->data();
			ptr[unfoldIndex(idx, res->shape())] = getValue<uint8_t>(idx, x);
		}
		else
			throw EtError("Cannot realize such dtype");
	});
	return res;
}
