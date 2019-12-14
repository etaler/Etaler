#include "Tensor.hpp"

#include <sstream>

using namespace et;
using std::size_t; //Surpress VSCode warnings

size_t g_print_threshold = 1000;
size_t g_truncate_size = 3;

template <typename T>
static size_t prettyPrintTensor(std::ostream& os, const T* arr, Shape shape, size_t depth, size_t maxDepth, size_t maxLength=0, bool truncate=false) noexcept
{
	auto floatToStr = [](float val) {
		std::stringstream ss;
		ss << val;
		return ss.str();
	};

	//If at the first dimention
	if(depth == 0) {
		//Calculatet the max character of printing a single element needs
		for(int i=0;i<shape.volume();i++)
			maxLength = std::max(maxLength, floatToStr(arr[i]).size());
	}

	std::string truncate_symbol = "....";

	//If at the the last dimention, print the content of the tensor
	if(shape.size() == 1) {
		os << "{ ";
		intmax_t size = shape[0];
		intmax_t max_line_content = intmax_t((80-depth*2-4)/(maxLength+2));

		//Print the full content
		if(size <= max_line_content || !truncate) {
			for(intmax_t i=0;i<size;i++) {
				std::string str = floatToStr(arr[i]);
				size_t padding_len = maxLength - str.size();
				os << str << std::string(padding_len, ' ') << (i==size-1 ? "" : ", ");
			}
		}
		//Print the truncated version. ex: {1, 1, 1, ... 1, 1, 1}
		else {
			//The first half
			for(intmax_t i=0;i<max_line_content/2;i++) {
				std::string str = floatToStr(arr[i]);
				size_t padding_len = maxLength - str.size();
				os << str << std::string(padding_len, ' ') << ", ";
			}

			//Seperator
			os << truncate_symbol << ", ";

			//The second half
			for(intmax_t i=size-max_line_content/2;i<size;i++) {
				std::string str = floatToStr(arr[i]);
				size_t padding_len = maxLength - str.size();
				os << str << std::string(padding_len, ' ') << (i==size-1 ? "" : ", ");
			}
		}

		os << "}";
		return 1;
	}

	intmax_t size = shape[0];
	shape.erase(shape.begin());
	intmax_t vol = shape.volume();

	size_t val = 0;
	os << "{";

	if(size < 2*intmax_t(g_truncate_size) || !truncate) {
		//The full version
		for(intmax_t i=0;i<size;i++) {
			//Print the data recursivelly
			val = prettyPrintTensor(os, arr+i*vol, shape, depth+1, maxDepth, maxLength, truncate);
			if(i != size-1)
				os << ", " << std::string(val, '\n') << (i==size-1 ? std::string("") : std::string(maxDepth-val, ' '));
		}
	}
	else {
		//The first half
		for(intmax_t i=0;i<intmax_t(g_truncate_size);i++) {
			//Print the data recursivelly
			val = prettyPrintTensor(os, arr+i*vol, shape, depth+1, maxDepth, maxLength, truncate);
			if(i != size-1)
				os << ", " << std::string(val, '\n') << std::string(maxDepth-val, ' ');
		}

		//seperator
		os << truncate_symbol << '\n' << std::string(maxDepth-val, ' ');

		//The second half
		for(intmax_t i=size-intmax_t(g_truncate_size);i<size;i++) {
			//Print the data recursivelly
			val = prettyPrintTensor(os, arr+i*vol, shape, depth+1, maxDepth, maxLength, truncate);
			if(i != size-1)
				os << ", " << std::string(val, '\n') << (i==size-1 ? std::string("") : std::string(maxDepth-val, ' '));
		}
	}
	os << "}";

	return val+1;//return the current depth from the back
}

static void printNDArray(std::ostream& os, const void* ptr, Shape shape, DType dtype)
{
	bool truncate = size_t(shape.volume()) > g_print_threshold;
	if(dtype == DType::Float)
		prettyPrintTensor(os, (float*)ptr, shape, 0, shape.size(), 0, truncate);
	else if(dtype == DType::Int32)
		prettyPrintTensor(os, (int32_t*)ptr, shape, 0, shape.size(), 0, truncate);
	else if(dtype == DType::Bool)
		prettyPrintTensor(os, (bool*)ptr, shape, 0, shape.size(), 0, truncate);
	else if(dtype == DType::Half)
		prettyPrintTensor(os, (half*)ptr, shape, 0, shape.size(), 0, truncate);
	else
		throw EtError("Printing tensor of this type is not supported.");
}

std::ostream& et::operator<< (std::ostream& os, const Tensor& t)
{
	if(t.pimpl() == nullptr) {
		os << "{}";
		return os;
	}

	Tensor q = realize(t);

	const void* ptr = q.data();
	if(ptr == nullptr) {
		void* buffer = malloc(q.size()*dtypeToSize(q.dtype()));
		q.backend()->copyToHost(q.pimpl(), buffer);
		printNDArray(os, buffer, q.shape(), q.dtype());
		free(buffer);
	}
	else {
		printNDArray(os, ptr, q.shape(), q.dtype());
	}

	return os;
}

std::string et::to_string(const Tensor& t)
{
	std::stringstream ss;
	ss << t;
	return ss.str();
}

Tensor Tensor::to(Backend* dest_backend) const
{
	if(pimpl()->iscontiguous() == false)
		return realize().to(dest_backend);
	return dest_backend->from(pimpl());
}

bool Tensor::isSame(const Tensor& other) const
{
	if(dtype() != other.dtype())
		return false;
	if(shape() != other.shape())
		return false;

	//A hacky comparsion
	return (*this == other).sum().item<int32_t>() == (int32_t)size();
}

Tensor Tensor::view(svector<Range> ranges) const
{
	if(ranges.size() > dimentions())
		throw EtError("Cannot view a tensor of " + std::to_string(dimentions()) + " with " + std::to_string(ranges.size()) + " dimentions");

	while(ranges.size() != dimentions())
		ranges.push_back(all());

	auto resolve_index = [](intmax_t idx, intmax_t size) -> intmax_t {
		if(idx < 0)
			return size-idx;
		return idx;
	};

	auto is_index_valid = [](intmax_t idx, intmax_t size) -> bool {
		if(idx >= 0)
			return idx < size;
		return -idx <= size;
	};

	Shape result_shape;
	svector<intmax_t> offset;
	Shape viewed_strides = pimpl_->stride();
	offset.reserve(dimentions());

	assert(viewed_strides.size() == dimentions());

	for(size_t i=0;i<dimentions();i++) {
		const Range& r = ranges[i];
		intmax_t dim_size = shape()[i];

		intmax_t start = r.start().value_or(0);
		intmax_t stop = r.stop().value_or(dim_size);
		intmax_t step = r.step().value_or(1);

		// Indexing validations
		if(step == 0)
			throw EtError("Error: Step size is zero in dimension " + std::to_string(i));
		if(is_index_valid(start, dim_size) == false)
			throw EtError("Starting index " + std::to_string(start) + " is out of range in dimension " + std::to_string(i));
		if(is_index_valid(stop, dim_size+1) == false)
			throw EtError("Stopping index " + std::to_string(stop) + " is out of range in dimension " + std::to_string(i));

		intmax_t real_start = resolve_index(start, dim_size);
		intmax_t real_stop = resolve_index(stop, dim_size);
		intmax_t size = (real_stop - real_start - 1) / step + 1;

		if((real_stop - real_start) * step < 0)
			throw EtError("Step is going in the wrong direction. Will cause infinate loop");
		viewed_strides[i] *= step;

		offset.push_back(real_start);
		if(size != 1 || result_shape.empty() == false) //Ignore heading 1 dimentions
			result_shape.push_back(size);
	}

	//If all dims are 1, thus no shape. Give it a shape
	if(result_shape.empty() == true)
		result_shape.push_back(1);

	size_t initial_offset = unfold(offset, pimpl_->stride())+pimpl_->offset();
	return std::make_shared<TensorImpl>(pimpl_->buffer(), result_shape, viewed_strides, initial_offset);
}

Tensor et::zeros(const Shape& shape, DType dtype, Backend* backend)
{
	if(dtype == DType::Bool)
		return constant<uint8_t>(shape, 0, backend);
	else if(dtype == DType::Int32)
		return constant<int32_t>(shape, 0, backend);
	else if(dtype == DType::Float)
		return constant<float>(shape, 0, backend);
	else if(dtype == DType::Half)
		return constant<half>(shape, half(0), backend);
	else
		throw EtError("Cannot creatr a tensor of zeros of type " + to_ctype_string(dtype));
}

Tensor et::ones(const Shape& shape, DType dtype, Backend* backend)
{
	if(dtype == DType::Bool)
		return constant<uint8_t>(shape, 1, backend);
	else if(dtype == DType::Int32)
		return constant<int32_t>(shape, 1, backend);
	else if(dtype == DType::Float)
		return constant<float>(shape, 1, backend);
	else if(dtype == DType::Half)
		return constant<half>(shape, half(1), backend);
	else
		throw EtError("Cannot creatr a tensor of ones of type " + to_ctype_string(dtype));
}

Tensor Tensor::sum(std::optional<intmax_t> dim_id, DType dtype) const
{
	et_assert(dim_id.has_value() || dim_id.value_or(0) < (intmax_t)dimentions());

	// dim_id has no value means sum the entire tensor
	if(dim_id.has_value() == false)
		return backend()->sum(pimpl(), size(), dtype);

	intmax_t dim = dim_id.value();
	// negative index means counting from back
	dim = dim < 0 ? dimentions() + dim : dim;
	if(dim >= (intmax_t)dimentions() || dim < 0)
		throw EtError("Dimension " + std::to_string(dim_id.value()) + " is out of range.");

	Shape final_shape = shape();
	final_shape.erase(final_shape.begin() + dim);
	intmax_t sum_size = shape()[dim];
	Shape result_shape = shape();
	result_shape[dim] = result_shape[result_shape.size()-1];
	result_shape.pop_back();

	if(size_t(dim) == dimentions()-1) { //Special, optimized case for the last dim
		Tensor res = backend()->sum(pimpl(), shape().back(), dtype);
		res.resize(final_shape);
		return res;
	}

	Tensor res = backend()->sum(swapaxis(dimentions()-1, dim).realize().pimpl(), sum_size, dtype);
	res.resize(result_shape);
	return res.swapaxis(res.shape().size()-1, dim).reshape(final_shape);
}

Tensor et::sum(const Tensor& x, std::optional<intmax_t> dim, DType dtype)
{
	return x.sum(dim, dtype);
}

Tensor et::cat(const svector<Tensor>& tensors, intmax_t dim)
{
	if(tensors.size() == 0)
		throw EtError("trying to concatenate 0 tensors together");

	// Check the tensors can be cated
	auto base_shape = tensors[0].shape();
	auto base_dtype = tensors[0].dtype();
	auto base_backend = tensors[0].backend();
	for(const auto& t : tensors) {
		if((intmax_t)t.dimentions() <= dim) {
			throw EtError("Requesting to concat along dim="+std::to_string(dim)
				+", but tensor is "+std::to_string(t.dimentions())+"D.");
		}

		if(base_dtype != t.dtype())
			throw EtError("DType mismatch when concatenate.");
		
		if(base_backend != t.backend())
			throw EtError("Backend mismatch when concatenate.");

		auto shape = t.shape();
		assert((intmax_t)shape.size() > dim);
		shape[dim] = base_shape[dim];
		if(shape != base_shape)
			throw EtError("Tensors must have the same shape along all axises besides the concatenating axis.");
	}

	Shape res_shape = base_shape;
	res_shape[dim] = std::accumulate(tensors.begin(), tensors.end(), intmax_t{0}
		, [&](intmax_t a, const auto& t){return a+t.shape()[dim];});
	Tensor res = Tensor(res_shape, base_dtype, base_backend);
	
	intmax_t pos = 0;
	svector<Range> ranges;
	for(size_t i=0;i<res_shape.size();i++)
		ranges.push_back(all());

	for(const auto& t : tensors) {
		ranges[dim] = Range(pos, pos+t.shape()[dim]);
		res.view(ranges) = t;
		pos = pos + t.shape()[dim];
	}

	return res;
}

Tensor Tensor::copy() const
{
	if(iscontiguous() == true)
		return backend()->copy(pimpl());
	return realize().copy();
}

inline bool brodcastable(Shape a, Shape b)
{
	if(a.size() == 0 || b.size() == 0)
		return false;

	size_t min = std::min(a.size(), b.size());

	for(size_t i=0;i<min;i++) {
		intmax_t s1 = *(a.rbegin()+i);
		intmax_t s2 = *(b.rbegin()+i);
		if(s1 == 1 || s2 == 1)
			continue;
		if(s1 != s2)
			return false;
	}

	return true;
}

inline Shape brodcast_result_shape(Shape a, Shape b)
{
	size_t max = std::max(a.size(), b.size());
	a = leftpad(a, max);
	b = leftpad(b, max);
	assert(a.size() == b.size());
	assert(a.size() == max);

	Shape s;
	for(size_t i=0;i<max;i++)
		s.push_back(std::max(a[i], b[i]));
	return s;
}

Tensor et::brodcast_to(const Tensor& t, Shape s)
{
	et_assert(s.size() >= t.dimentions());
	Shape stride = leftpad(t.stride(), s.size(), 0);
	Shape shape = leftpad(t.shape(), s.size(), 0);
	for(size_t i=0;i<s.size();i++) {
		if(shape[i] != s[i])
			stride[i] = 0;
	}
	return std::make_shared<TensorImpl>(t.shared_pimpl()->buffer(), s, stride, t.pimpl()->offset());
}

std::pair<Tensor, Tensor> et::brodcast_tensors(const Tensor& a, const Tensor& b)
{
	if(brodcastable(a.shape(), b.shape()) == false)
		throw EtError("Cannot brodcast " + to_string(a.shape()) + " and " + to_string(b.shape()) + " together.");

	Shape result_shape = brodcast_result_shape(a.shape(), b.shape());

	return {brodcast_to(a, result_shape), brodcast_to(b, result_shape)};
}

std::pair<Tensor, Tensor> Tensor::brodcast(const Tensor& other) const
{
	return brodcast_tensors(*this, other);
}
