#include "CPUBackend.hpp"
#include "Etaler/Core/Views.hpp"
#include "Etaler/Core/Random.hpp"
#include "Etaler/Core/TypeList.hpp"

#include <numeric>
#include <cmath>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_sort.h>
#include <tbb/parallel_reduce.h>

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


void* CPUBuffer::data() const
{
	assert(dtype() != DType::Unknown);
	return std::visit([](const auto& v){return (void*)v;}, storage_);
}

using DefaultTypeList = type_list_t<int32_t, float, bool, half>;

template <typename TypeList = DefaultTypeList, typename Func = void>
inline void dispatch(DType dtype, Func f)
{
	static_assert(std::is_same_v<Func, void> == false); //void is just a dummy value
	if constexpr(std::is_same_v<TypeList, null_t> == false) {
		using T = typename TypeList::head;
		if(typeToDType<T>() == dtype) {
			f(T());
			return;
		}
		dispatch<typename TypeList::tail, Func>(dtype, f);
	}
	else
		throw EtError("Cannot dispatch such dtype: " + to_ctype_string(dtype));
}

template <typename TL1 = DefaultTypeList, typename TL2 = DefaultTypeList, typename Func = void>
inline void dispatch2d(DType t1, DType t2, Func f)
{
	dispatch<TL1>(t1, [&](auto v1){
		using T1 = decltype(v1);
		dispatch<TL2>(t2, [&](auto v2){
			using T2 = decltype(v2);
			f(T1(), T2());
		});
	});
}

namespace et::detail
{
template <typename PermType>
static std::shared_ptr<TensorImpl> cellActivity(const TensorImpl* x, const TensorImpl* connections, const TensorImpl* permeances
	, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse, CPUBackend* backend)
{
	//Checks the input are sane
	requireProperties(x, backend, DType::Bool, IsPlain());
	requireProperties(connections, backend, DType::Int32, IsPlain());
	requireProperties(permeances, backend, typeToDType<PermType>(), IsPlain());
	et_assert(connections->shape() == permeances->shape());
	et_assert(connections->dimentions() >= 2);

	Shape s = connections->shape();
	s.pop_back();
	auto y = backend->createTensor(s, DType::Int32);


	const bool* input = (const bool*)x->data();
	const int32_t* synapses = (const int32_t*)connections->data();
	const PermType* synapse_strengths = (PermType*)permeances->data();
	int32_t* result = (int32_t*)y->data();

	size_t max_connections_per_cell = connections->shape().back();
	size_t num_cells = connections->size()/max_connections_per_cell;


	size_t block_size = std::min(size_t(128), (size_t)num_cells);
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

				PermType strength = synapse_strengths[index];
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

template <typename PermType>
void learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections, TensorImpl* permeances
	, float perm_inc, float perm_dec, bool has_unconnected_synapse, CPUBackend* backend)
{
	requireProperties(x, backend, DType::Bool, IsPlain());
	requireProperties(learn, backend, DType::Bool, IsPlain());
	requireProperties(connections, backend, DType::Int32, IsPlain());
	requireProperties(permeances, backend, typeToDType<PermType>(), IsPlain());

	et_assert(connections->shape() == permeances->shape());

	const bool* input = (const bool*)x->data();
	const bool* learning = (const bool*)learn->data();
	const int32_t* synapses = (const int32_t*)connections->data();
	PermType* synapse_strengths = (PermType*)permeances->data();

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

			PermType& perm = synapse_strengths[idx];
			if(input[connection] == true)
				perm += perm_inc;
			else
				perm -= perm_dec;

			perm = std::clamp(perm, PermType(0), PermType(1));
		}
	});
}

template <typename PermType>
void sortSynapse(TensorImpl* connections, TensorImpl* permeances, CPUBackend* backend)
{
	requireProperties(connections, backend, DType::Int32, IsPlain());
	requireProperties(permeances, backend, typeToDType<PermType>(), IsPlain());
	et_assert(connections->shape() == permeances->shape());

	size_t max_synapse_per_cell = connections->shape().back();
	size_t num_cells = connections->size()/max_synapse_per_cell;

	uint32_t* conns = (uint32_t*)connections->data(); //HACK: -1s should be at the end of the arrays.
	PermType* perms = (PermType*)permeances->data();

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

template <typename PermType>
void growSynapses(const TensorImpl* x, const TensorImpl* y, TensorImpl* connections
	, TensorImpl* permeances, float initial_perm, CPUBackend* backend)
{
	requireProperties(x, backend, DType::Bool, IsPlain());
	requireProperties(y, backend, DType::Bool, IsPlain());
	requireProperties(connections, backend, DType::Int32, IsPlain());
	requireProperties(permeances, backend, typeToDType<PermType>(), IsPlain());

	et_assert(connections->shape() == permeances->shape());
	Shape s = connections->shape();
	s.pop_back();
	et_assert(s == y->shape());

	size_t max_synapses_per_cell = connections->shape().back();
	size_t input_cell_count = x->size();

	const bool* in = (const bool*) x->data();
	const bool* out = (const bool*) y->data();
	int32_t* conns = (int32_t*)connections->data();
	PermType* perms = (PermType*)permeances->data();

	std::vector<uint32_t> on_bits;
	on_bits.reserve(input_cell_count*0.1);
	for(size_t i=0;i<input_cell_count;i++) {
		if(in[i] == true)
			on_bits.push_back(i);
	}

	size_t block_size = std::min(size_t(16), (size_t)y->shape().back());
	tbb::parallel_for(tbb::blocked_range<size_t>(size_t(0), y->size(), block_size), [&](const auto& r) {
		for(size_t i=r.begin();i!=r.end();i++) {
			if(out[i] == 0)
				continue;

			uint32_t* synapses = (uint32_t*)conns+i*max_synapses_per_cell;
			PermType* strengths = perms+i*max_synapses_per_cell;
			uint32_t* end = synapses+max_synapses_per_cell;

			if(synapses[max_synapses_per_cell-1] != uint32_t(-1)) //If there is no space for new synapse. Ignore
				continue;

			uint32_t* it = std::lower_bound(synapses, end, uint32_t(-1));
			size_t used_space = it - synapses;

			size_t write_idx = it - synapses;
			size_t read_idx = 0;

			for(size_t j=0;write_idx!=max_synapses_per_cell && j < on_bits.size();j++) {
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

template <typename PermType>
void decaySynapses(TensorImpl* connections, TensorImpl* permeances, float threshold, CPUBackend* backend)
{
	requireProperties(connections, backend, DType::Int32, IsPlain());
	requireProperties(permeances, backend, typeToDType<PermType>(), IsPlain());
	et_assert(connections->shape() == permeances->shape());

	PermType* perms = (PermType*)permeances->data();
	uint32_t* conns = (uint32_t*)connections->data();

	size_t max_synapses_per_cell = connections->shape().back();
	size_t input_cell_count = connections->size()/max_synapses_per_cell;

	tbb::parallel_for(size_t(0), input_cell_count, [&](size_t i) {
		uint32_t* synapses = (uint32_t*)conns+i*max_synapses_per_cell;
		PermType* strengths = perms+i*max_synapses_per_cell;
		uint32_t* end = synapses+max_synapses_per_cell;

		uint32_t* it = std::lower_bound(synapses, end, uint32_t(-1));
		size_t used_space = it - synapses;

		for(size_t j=0;j<used_space;j++) {
			if(strengths[j] < threshold)
				synapses[j] = uint32_t(-1);
		}

		std::vector<size_t> sort_indices(used_space);
		std::iota(sort_indices.begin(), sort_indices.begin()+used_space, 0);
		std::sort(sort_indices.begin(), sort_indices.begin()+used_space,
			[&](size_t i, size_t j)->bool {
				return ((uint32_t*)synapses)[i] < ((uint32_t*)synapses)[j];
			});
		apply_permutation_in_place(synapses, synapses+used_space, sort_indices);
		apply_permutation_in_place(strengths, strengths+used_space, sort_indices);
	});
}

}

std::shared_ptr<TensorImpl> CPUBackend::cellActivity(const TensorImpl* x, const TensorImpl* connections, const TensorImpl* permeances
	, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse)
{
	std::shared_ptr<TensorImpl> res;
	dispatch<type_list_t<float, half>>(permeances->dtype(), [&](auto v){
		res = detail::cellActivity<decltype(v)>(x, connections, permeances, connected_permeance, active_threshold, has_unconnected_synapse, this);
	});
	return res;
}

void CPUBackend::learnCorrilation(const TensorImpl* x, const TensorImpl* learn, const TensorImpl* connections, TensorImpl* permeances
	, float perm_inc, float perm_dec, bool has_unconnected_synapse)
{
	dispatch<type_list_t<float, half>>(permeances->dtype(), [&](auto v){
		detail::learnCorrilation<decltype(v)>(x, learn, connections, permeances, perm_inc, perm_dec, has_unconnected_synapse, this);
	});
}

std::shared_ptr<TensorImpl> CPUBackend::globalInhibition(const TensorImpl* x, float fraction)
{
	requireProperties(x, this, DType::Int32, IsPlain());

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

	//If we have a empty input
	if(v.size() == 0)
		return y;

	tbb::parallel_sort(v.begin(), v.end(), [](const auto& a, const auto&b){return a.first > b.first;});

	for(size_t i=0;i<y->size();i++)
		output[i] = false;
	size_t accept_index = std::min((target_size==0? 0 : target_size-1), v.size()-1);
	int32_t min_accept_val = v[accept_index].first;
	auto bound_end = std::upper_bound(v.begin()+accept_index, v.end(), min_accept_val, [](const auto& a, const auto& b){return a > b.first;});

	for(auto it=v.begin();it!=bound_end;++it)
		output[it->second] = true;
	return y;
}


template <typename To, typename From>
static auto castData(const From* ptr, size_t n)
{
	// Deal with the special case of bool
	if constexpr(std::is_same_v<std::decay_t<To>, bool>) {
		std::vector<uint8_t> res(n);
		std::transform(ptr, ptr+n, res.begin(), [](auto a){ return (bool)a; });
		return res;
	}
	else
		return std::vector<To>(ptr, ptr+n);
}

std::shared_ptr<TensorImpl> CPUBackend::cast(const TensorImpl* x, DType toType)
{
	requireProperties(x, this, IsPlain());
	auto res = createTensor(x->shape(), toType);
	dispatch(toType, [&](auto v0){
		using ToType = decltype(v0);
		dispatch(x->dtype(), [&](auto v1){
			using FromType = decltype(v1);
			auto casted_data = castData<ToType>((FromType*)x->data(), x->size());
			static_assert(sizeof(typename decltype(casted_data)::value_type) == sizeof(ToType));
			memcpy(res->data(), casted_data.data(), casted_data.size()*sizeof(ToType));
		});
		
	});
	return res;
}

void CPUBackend::copyToHost(const TensorImpl* t, void* ptr)
{
	requireProperties(t, this, IsPlain());
	memcpy(ptr, t->data(), t->size()*dtypeToSize(t->dtype()));
}

std::shared_ptr<TensorImpl> CPUBackend::copy(const TensorImpl* x)
{
	requireProperties(x, this, IsPlain());
	return createTensor(x->shape(), x->dtype(), x->data());
}

void CPUBackend::sortSynapse(TensorImpl* connections, TensorImpl* permeances)
{
	dispatch<type_list_t<float, half>>(permeances->dtype(), [&](auto v) {
		detail::sortSynapse<decltype(v)>(connections, permeances, this);
	});
}

std::shared_ptr<TensorImpl> CPUBackend::burst(const TensorImpl* x, const TensorImpl* s)
{
	requireProperties(x, this, DType::Bool, IsPlain());
	requireProperties(s, this, DType::Bool, IsPlain());

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
		else {
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
	requireProperties(x, this, DType::Bool, IsPlain());

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
	dispatch<type_list_t<float, half>>(permeances->dtype(), [&](auto v) {
		detail::growSynapses<decltype(v)>(x, y, connections, permeances, initial_perm, this);
	});
}

template <typename T>
const T* getPtrToValue(size_t parent_idx, const TensorImpl* t)
{
	// Optimized case for contnigous input
	if(t->iscontiguous())
		return (const T*)t->data()+t->offset()+parent_idx;
	
	Shape s = foldIndex(parent_idx, t->shape());
	s = Shape(t->stride().size()-s.size(), 0) + s;
	size_t offset = t->offset() + unfold(s, t->stride());
	return ((const T*)t->data())+offset;
}

template <typename T2, typename T1>
void write(T2* ptr, T1 v)
{
	*ptr = v;
}

template <typename Op>
static std::shared_ptr<TensorImpl> uniaryOp(const TensorImpl* src, Op op)
{
	std::shared_ptr<TensorImpl> dest;
	dispatch(src->dtype(), [&](auto v){
		using T = decltype(v);
		using ResType = std::invoke_result_t<Op, T>;
		//We don't have support to double percition now. Cast it to float
		using StoreType = typename std::conditional_t<std::is_same_v<ResType, bool>, bool
			, typename std::conditional_t<std::is_same_v<T, half>, half
			, typename std::conditional_t<std::is_same_v<ResType, double>, float, ResType>>>;
		dest = src->backend()->createTensor(src->shape(), typeToDType<StoreType>());
		tbb::parallel_for(size_t(0), src->size(), [&](size_t i) {
			auto ptr = getPtrToValue<T>(i, src);
			auto res = op(*ptr);

			reinterpret_cast<StoreType*>(dest->data())[i] = res;
		});
	});

	et_assert((bool)dest);
	return dest;
}

template <typename Op>
static std::shared_ptr<TensorImpl> binaryOp(const TensorImpl* src, const TensorImpl* src2, Op op)
{
	std::shared_ptr<TensorImpl> dest;
	et_assert(src->shape() == src2->shape());

	dispatch(src->dtype(), [&](auto v){
		using T1 = decltype(v);
		dispatch(src2->dtype(), [&](auto v){
			using T2 = decltype(v);
			using ResType = std::invoke_result_t<Op, T1, T2>;
			//We don't have support to double percition now. Cast it to float
			using StoreType = typename std::conditional<std::is_same<ResType, double>::value, float, ResType>::type;
			dest = src->backend()->createTensor(src->shape(), typeToDType<StoreType>());

			tbb::parallel_for(size_t(0), src->size(), [&](size_t i) {
				auto ptr = getPtrToValue<T1>(i, src);
				auto ptr2 = getPtrToValue<T2>(i, src2);
				auto res = op(*ptr, *ptr2);

				reinterpret_cast<StoreType*>(dest->data())[i] = res;
			});
		});
	});

	et_assert((bool)dest);
	return dest;
}

std::shared_ptr<TensorImpl> CPUBackend::realize(const TensorImpl* x)
{
	requireProperties(x, this);
	et_assert(x->data() != nullptr);
	auto res = createTensor(x->shape(), x->dtype());

	dispatch(x->dtype(), [&](auto v){
		using T = decltype(v);
		for(size_t i=0;i<x->size();i++) {
			auto ptr = (T*)res->data();
			ptr[i] = *getPtrToValue<T>(i, x);
		}
	});
	return res;
}


void CPUBackend::assign(TensorImpl* dest, const TensorImpl* src)
{
	requireProperties(dest, this);
	requireProperties(src, this);

	if(dest->shape() != src->shape())
		throw EtError("Shape mismatch in tensor assignment. Shape "
			+ to_string(dest->shape()) + " and " + to_string(src->shape()));

	if(dest->dtype() != src->dtype())
		assign(dest, cast(src, dest->dtype()).get());

	dispatch(dest->dtype(), [&](auto v) {
		using T = decltype(v);

		//Parallelize if the problem is big enought
		if(dest->size() > 2000) {
			tbb::parallel_for(size_t(0), dest->size(), [&](size_t i) {
				auto s = (T*)getPtrToValue<T>(i, src);
				auto ptr = (T*)getPtrToValue<T>(i, dest);
				*ptr = *s;
			});
		}
		else {
			for(size_t i=0;i<dest->size();i++) {
				auto s = (T*)getPtrToValue<T>(i, src);
				auto ptr = (T*)getPtrToValue<T>(i, dest);
				*ptr = *s;
			}
		}
	});
}

std::shared_ptr<TensorImpl> CPUBackend::sum(const TensorImpl* x, size_t chunk_size, DType dtype)
{
	requireProperties(x, this, IsPlain());
	et_assert(x->size() % chunk_size == 0);

	DType result_dtype = dtype;

	if(dtype == DType::Unknown) {
		result_dtype = [x](){
			DType dtype = x->dtype();
			if(dtype == DType::Bool || dtype == DType::Int32)
				return DType::Int32;
			else if(dtype == DType::Half)
				return DType::Half;
			else
				return DType::Float;
		}();
	}

	size_t result_size = x->size()/chunk_size;
	auto res = createTensor({intmax_t(result_size)}, result_dtype);

	// Optimized case for summing everything
	if(result_size == 1) {
		dispatch2d(x->dtype(), result_dtype, [&](auto v1, auto v2) {
			using T = decltype(v1);
			auto in = (const T*)x->data();
			using ResType = decltype(v2);
			auto ptr = (ResType*) res->data();
			*ptr = tbb::parallel_reduce(tbb::blocked_range(in, in+x->size()), ResType(0)
				, [](const auto& r, ResType init){
					return std::accumulate(r.begin(), r.end(), init);
				},
				[](auto x, auto y) {
					return x + y;
				});
		});
	}
	else {
		dispatch2d(x->dtype(), result_dtype, [&](auto v1, auto v2) {
			using T = decltype(v1);
			auto in = (const T*)x->data();
			using ResType = decltype(v2);
			auto ptr = (ResType*) res->data();
			tbb::parallel_for(size_t(0), size_t(x->size()/chunk_size), [&](size_t i) {
				size_t offset = i*chunk_size;
				ResType s = std::accumulate(in+offset, in+offset+chunk_size, ResType(0));
				ptr[i] = s;
			});
		});
	}

	return res;
}

void CPUBackend::decaySynapses(TensorImpl* connections, TensorImpl* permeances, float threshold)
{
	dispatch<type_list_t<float, half>>(permeances->dtype(), [&](auto v) {
		detail::decaySynapses<decltype(v)>(connections, permeances, threshold, this);
	});
}

std::shared_ptr<TensorImpl> CPUBackend::abs(const TensorImpl* x)
{
	return uniaryOp(x, [](auto v){return std::abs(v);});
}

std::shared_ptr<TensorImpl> CPUBackend::exp(const TensorImpl* x)
{
	return uniaryOp(x, [](auto v){return std::exp(v);});
}

std::shared_ptr<TensorImpl> CPUBackend::negate(const TensorImpl* x)
{
	return uniaryOp(x, [](auto v){return -v;});
}

std::shared_ptr<TensorImpl> CPUBackend::inverse(const TensorImpl* x)
{
	return uniaryOp(x, [](auto v){return 1.f/v;});
}

std::shared_ptr<TensorImpl> CPUBackend::log(const TensorImpl* x)
{
	return uniaryOp(x, [](auto v){return std::log(v);});
}

std::shared_ptr<TensorImpl> CPUBackend::logical_not(const TensorImpl* x)
{
	return uniaryOp(x, [](auto v){return !((bool)v);});
}

std::shared_ptr<TensorImpl> CPUBackend::add(const TensorImpl* x1, const TensorImpl* x2)
{
	return binaryOp(x1, x2, [](auto a, auto b) {return a+b;});
}
std::shared_ptr<TensorImpl> CPUBackend::subtract(const TensorImpl* x1, const TensorImpl* x2)
{
	return binaryOp(x1, x2, [](auto a, auto b) {return a-b;});
}
std::shared_ptr<TensorImpl> CPUBackend::mul(const TensorImpl* x1, const TensorImpl* x2)
{
	return binaryOp(x1, x2, [](auto a, auto b) {return a*b;});
}
std::shared_ptr<TensorImpl> CPUBackend::div(const TensorImpl* x1, const TensorImpl* x2)
{
	return binaryOp(x1, x2, [](auto a, auto b) {return a/b;});
}

std::shared_ptr<TensorImpl> CPUBackend::equal(const TensorImpl* x1, const TensorImpl* x2)
{
	return binaryOp(x1, x2, [](auto a, auto b) {return a==b;});
}
std::shared_ptr<TensorImpl> CPUBackend::greater(const TensorImpl* x1, const TensorImpl* x2)
{
	return binaryOp(x1, x2, [](auto a, auto b) {return a>b;});
}
std::shared_ptr<TensorImpl> CPUBackend::lesser(const TensorImpl* x1, const TensorImpl* x2)
{
	return binaryOp(x1, x2, [](auto a, auto b) {return a<b;});
}
std::shared_ptr<TensorImpl> CPUBackend::logical_and(const TensorImpl* x1, const TensorImpl* x2)
{
	return binaryOp(x1, x2, [](auto a, auto b) {return a&&b;});
}
std::shared_ptr<TensorImpl> CPUBackend::logical_or(const TensorImpl* x1, const TensorImpl* x2)
{
	return binaryOp(x1, x2, [](auto a, auto b) {return a||b;});
}

std::shared_ptr<TensorImpl> CPUBackend::from(const TensorImpl* x)
{
	const void* ptr = x->data();
	if(ptr != nullptr)
		return createTensor(x->shape(), x->dtype(), ptr);

	void* buffer = malloc(x->size()*dtypeToSize(x->dtype()));
	x->backend()->copyToHost(x, buffer);
	auto res = createTensor(x->shape(), x->dtype(), buffer);
	free(buffer);
	return res;
}
