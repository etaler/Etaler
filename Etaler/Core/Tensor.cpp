#include "Tensor.hpp"

#include <sstream>

using namespace et;
using std::size_t; //Surpress VSCode warnings

template <typename T>
static size_t prettyPrintTensor(std::ostream& os, const T* arr, Shape shape, size_t depth, size_t maxDepth, size_t maxLength=0) noexcept
{
	auto floatToStr = [](float val) {
		std::stringstream ss;
		ss << val;
		return ss.str();
	};

	//If at the first dimention
	if(depth == 0) {
		//Calculatet the max length of the line
		for(int i=0;i<shape.volume();i++)
			maxLength = std::max(maxLength, floatToStr(arr[i]).size());
	}

	//If at the the last dimention, print the content of the tensor
	if(shape.size() == 1) {
		os << "{ ";
		intmax_t size = shape[0];

		for(intmax_t i=0;i<size;i++) {
			std::string str = floatToStr(arr[i]);
			size_t padding_len = maxLength - str.size();
			os << str << std::string(padding_len, ' ') << (i==size-1 ? "" : ", ");
		}

		os << "}";
		return 1;
	}

	intmax_t size = shape[0];
	shape.erase(shape.begin());
	intmax_t vol = shape.volume();

	size_t val = 0;
	//TODO: Print spaces to align the text
	os << "{";
	for(intmax_t i=0;i<size;i++) {
		//Print the data recursivelly
		val = prettyPrintTensor(os, arr+i*vol, shape, depth+1, maxDepth, maxLength);
		if(i != size-1)
			os << ", " << std::string(val, '\n') << std::string(maxDepth-val, ' ');

	}
	os << "}";

	return val+1;//return the current depth from the back
}

static void printNDArray(std::ostream& os, const void* ptr, Shape shape, DType dtype)
{
	if(dtype == DType::Float)
		prettyPrintTensor(os, (float*)ptr, shape, 0, shape.size());
	else if(dtype == DType::Int32)
		prettyPrintTensor(os, (int32_t*)ptr, shape, 0, shape.size());
	else if(dtype == DType::Bool)
		prettyPrintTensor(os, (bool*)ptr, shape, 0, shape.size());
	else
		throw EtError("Printing tensor of this type is not supported.");
}

std::ostream& et::operator<< (std::ostream& os, const Tensor& t)
{
	if(t.pimpl() == nullptr) {
		os << "{}";
		return os;
	}

	Tensor q = attempt_realize(t);

	const void* ptr = q.data();
	if(ptr == nullptr) {
		void* buffer = malloc(q.size()*dtypeToSize(q.dtype()));
		q.backend()->copyToHost(q.pimpl(), buffer);
		printNDArray(os, buffer, q.shape(), q.dtype());
		free(buffer);
	}
	else
		printNDArray(os, ptr, q.shape(), q.dtype());

	return os;
}

Tensor Tensor::to(Backend* dest_backend) const
{
	const void* ptr = data();
	if(ptr != nullptr)
		return dest_backend->createTensor(shape(), dtype(), ptr);

	//Use the main memory if direct access not avliable
	void* buffer = malloc(size()*dtypeToSize(dtype()));
	backend()->copyToHost(pimpl(), buffer);
	Tensor res = dest_backend->createTensor(shape(), dtype(), buffer);
	free(buffer);
	return res;
}

template <typename T>
bool compareTensor(const Tensor& a, const Tensor& b)
{
	auto a1 = a.toHost<T>();
	auto a2 = b.toHost<T>();
	return memcmp(a1.data(), a2.data(), a1.size()*sizeof(T)) == 0;
}

bool Tensor::isSame(const Tensor& other) const
{
	if(dtype() != other.dtype())
		return false;
	if(shape() != other.shape())
		return false;

	if(dtype() == DType::Float)
		return compareTensor<float>(*this, other);
	else if(dtype() == DType::Int32)
		return compareTensor<int32_t>(*this, other);
	else if(dtype() == DType::Bool)
		return compareTensor<uint8_t>(*this, other);
	et_assert(dtype() != DType::Unknown);
	return false;
}

Tensor et::zeros(const Shape& shape, DType dtype, Backend* backend)
{
	if(dtype == DType::Bool)
		return constant<uint8_t>(shape, 0, backend);
	else if(dtype == DType::Int32)
		return constant<int32_t>(shape, 0, backend);
	else if(dtype == DType::Float)
		return constant<float>(shape, 0, backend);
	else
		throw EtError("Cannot creatr a tensor of ones of type " + to_ctype_string(dtype));
}

Tensor et::ones(const Shape& shape, DType dtype, Backend* backend)
{
	if(dtype == DType::Bool)
		return constant<uint8_t>(shape, 1, backend);
	else if(dtype == DType::Int32)
		return constant<int32_t>(shape, 1, backend);
	else if(dtype == DType::Float)
		return constant<float>(shape, 1, backend);
	else
		throw EtError("Cannot creatr a tensor of zeros of type " + to_ctype_string(dtype));
}
