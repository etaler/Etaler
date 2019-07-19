#include "CPUBackend.hpp"
#include "Etaler/Core/Views.hpp"
#include "Etaler/Core/Random.hpp"
#include "Etaler/Core/TypeList.hpp"

#include <numeric>
#include <cmath>

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


void* CPUBuffer::data() const
{
	assert(dtype() != DType::Unknown);
	return std::visit([](const auto& v){return (void*)v;}, storage_);
}

std::shared_ptr<TensorImpl> CPUBackend::cellActivity(const TensorImpl* x, const TensorImpl* connections, const TensorImpl* permeances
	, float connected_permeance, size_t active_threshold, bool has_unconnected_synapse)
{
	//Checks the input are sane
	et_assert(x->backend() == this);
	et_assert(connections->backend() == this);
	et_assert(permeances->backend() == this);
	et_assert(x->iscontiguous());
	et_assert(connections->iscontiguous());
	et_assert(permeances->iscontiguous());

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
	et_assert(x->backend() == this);
	et_assert(connections->backend() == this);
	et_assert(permeances->backend() == this);
	et_assert(learn->backend() == this);
	et_assert(x->iscontiguous());
	et_assert(learn->iscontiguous());
	et_assert(connections->iscontiguous());
	et_assert(permeances->iscontiguous());

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
	et_assert(x->backend() == this);
	et_assert(x->iscontiguous());

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
static Ret run(const CPUBuffer& t, Op op)
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
	et_assert(x->backend() == this);
	et_assert(x->iscontiguous());
	const CPUBuffer* p = dynamic_cast<const CPUBuffer*>(x->buffer().get());
	const CPUBuffer& t = *p;
	return run<std::shared_ptr<TensorImpl>>(t, [&x, toType, this](const auto* ptr){
		if(toType == DType::Bool) {
			auto bool_vec = castData<bool>(ptr, x->size());
			std::vector<uint8_t> casted_data(bool_vec.begin(), bool_vec.end());
			return createTensor(x->shape(), toType, casted_data.data());
		}
		else if(toType == DType::Int32) {
			auto casted_data = castData<int32_t>(ptr, x->size());
			return createTensor(x->shape(), toType, casted_data.data());
		}
		else if(toType == DType::Float){
			auto casted_data = castData<float>(ptr, x->size());
			return createTensor(x->shape(), toType, casted_data.data());
		}
		else
			throw EtError("Cannot cast");
	});
}

void CPUBackend::copyToHost(const TensorImpl* t, void* ptr)
{
	et_assert(points_to<CPUBuffer>(t->buffer().get()));
	et_assert(t->iscontiguous());
	memcpy(ptr, t->data(), t->size()*dtypeToSize(t->dtype()));
}

std::shared_ptr<TensorImpl> CPUBackend::copy(const TensorImpl* x)
{
	et_assert(x->backend() == this);
	et_assert(x->iscontiguous());
	return createTensor(x->shape(), x->dtype(), x->data());
}

void CPUBackend::sortSynapse(TensorImpl* connections, TensorImpl* permeances)
{
	et_assert(connections->shape() == permeances->shape());
	et_assert(connections->backend() == this);
	et_assert(permeances->backend() == this);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);
	et_assert(connections->iscontiguous());
	et_assert(permeances->iscontiguous());

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

std::shared_ptr<TensorImpl> CPUBackend::burst(const TensorImpl* x, const TensorImpl* s)
{
	et_assert(x->backend() == this);
	et_assert(s->backend() == this);
	et_assert(x->dtype() == DType::Bool);
	et_assert(s->dtype() == DType::Bool);
	et_assert(x->iscontiguous());
	et_assert(s->iscontiguous());

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
	et_assert(x->backend() == this);
	et_assert(x->dtype() == DType::Bool);
	et_assert(x->iscontiguous());

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
	et_assert(x->backend() == this);
	et_assert(y->backend() == this);
	et_assert(connections->backend() == this);
	et_assert(permeances->backend() == this);
	et_assert(x->iscontiguous());
	et_assert(y->iscontiguous());
	et_assert(connections->iscontiguous());
	et_assert(permeances->iscontiguous());

	et_assert(x->dtype() == DType::Bool);
	et_assert(y->dtype() == DType::Bool);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);

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
			float* strengths = perms+i*max_synapses_per_cell;
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

template <typename T>
const T* getPtrToValue(size_t parent_idx, const TensorImpl* t)
{
	Shape s = foldIndex(parent_idx, t->shape());
	s = Shape(t->stride().size()-s.size(), 0) + s;
	size_t offset = t->offset() + unfold(s, t->stride());
	return ((const T*)t->data())+offset;
}

template <typename Func, typename TypeList = type_list_t<int32_t, float, bool, half>>
inline void dispatch(DType dtype, Func f)
{
	if constexpr(std::is_same_v<TypeList, null_t> == false) {
		using T = typename TypeList::head;
		if(typeToDType<T>() == dtype) {
			f(T());
			return;
		}
		dispatch<Func, typename TypeList::tail>(dtype, f);
	}
	else
		throw EtError("Cannot dispatch such dtype ID: " + (int)dtype + to_ctype_string(dtype));
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
		for(size_t i=0;i<src->size();i++) {
			auto ptr = getPtrToValue<T>(i, src);
			auto res = op(*ptr);
			using ResType = decltype(res);
			//We don't have support to double percition now. Cast it to float
			using StoreType = typename std::conditional_t<std::is_same_v<T, half>, half
				, typename std::conditional_t<std::is_same_v<ResType, double>, float, ResType>>;
			if(i == 0)
				dest = src->backend()->createTensor(src->shape(), typeToDType<StoreType>());

			reinterpret_cast<StoreType*>(dest->data())[i] = res;
		}
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
		using T = decltype(v);

		dispatch(src2->dtype(), [&](auto v){
			using T2 = decltype(v);

			for(size_t i=0;i<src->size();i++) {
				auto ptr = getPtrToValue<T>(i, src);
				auto ptr2 = getPtrToValue<T2>(i, src2);
				auto res = op(*ptr, *ptr2);
				using ResType = decltype(res);
				//We don't have support to double percition now. Cast it to float
				using StoreType = typename std::conditional<std::is_same<ResType, double>::value, float, ResType>::type;
				if(i == 0)
					dest = src->backend()->createTensor(src->shape(), typeToDType<StoreType>());

				reinterpret_cast<StoreType*>(dest->data())[i] = res;
			}
		});
	});

	et_assert((bool)dest);
	return dest;
}

std::shared_ptr<TensorImpl> CPUBackend::realize(const TensorImpl* x)
{
	et_assert(x->backend() == this);
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
	et_assert(dest->backend() == this);
	et_assert(src->backend() == this);

	if(dest->shape() != src->shape())
		throw EtError("Shape mismatch in tensor assignment. Shape "
			+ to_string(dest->shape()) + "and " + to_string(src->shape()));

	auto source = realize(src);

	if(dest->dtype() != src->dtype())
		source = cast(realize(source.get()).get(), dest->dtype());

	dispatch(dest->dtype(), [&](auto v) {
		using T = decltype(v);
		auto s = (T*)source->data();
		for(size_t i=0;i<dest->size();i++) {
			auto ptr = (T*)getPtrToValue<T>(i, dest);
			*ptr = s[i];
		}
	});
}

std::shared_ptr<TensorImpl> CPUBackend::sum(const TensorImpl* x, size_t chunk_size, DType dtype)
{
	et_assert(x->backend() == this);
	et_assert(x->size() % chunk_size == 0);
	et_assert(x->iscontiguous());

	DType result_dtype = dtype;

	if(dtype == DType::Unknown) {
		result_dtype = [x](){
			DType dtype = x->dtype();
			if(dtype == DType::Bool || dtype == DType::Int32)
				return DType::Int32;
			else
				return DType::Float;
		}();
	}

	auto res = createTensor({(intmax_t)(x->size()/chunk_size)}, result_dtype);

	dispatch(x->dtype(), [&](auto v){
		using T = decltype(v);
		auto in = (const T*)x->data();
		dispatch(result_dtype, [&](auto v) {
			using ResType = decltype(v);
			auto ptr = (ResType*) res->data();
			for(size_t i=0;i<x->size()/chunk_size;i++) {
				size_t offset = i*chunk_size;
				ResType s = std::accumulate(in+offset, in+offset+chunk_size, ResType(0));
				ptr[i] = s;
			}
		});
	});

	return res;
}

void CPUBackend::decaySynapses(TensorImpl* connections, TensorImpl* permeances, float threshold)
{
	et_assert(connections->shape() == permeances->shape());
	et_assert(connections->backend() == this);
	et_assert(permeances->backend() == this);
	et_assert(connections->dtype() == DType::Int32);
	et_assert(permeances->dtype() == DType::Float);
	et_assert(permeances->iscontiguous());
	et_assert(connections->iscontiguous());

	float* perms = (float*)permeances->data();
	uint32_t* conns = (uint32_t*)connections->data();

	size_t max_synapses_per_cell = connections->shape().back();
	size_t input_cell_count = connections->size()/max_synapses_per_cell;

	tbb::parallel_for(size_t(0), input_cell_count, [&](size_t i) {
		uint32_t* synapses = (uint32_t*)conns+i*max_synapses_per_cell;
		float* strengths = perms+i*max_synapses_per_cell;
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