#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Encoders/Category.hpp>
using namespace et;

#include <iostream>
#include <numeric>

//TODO: This should be computed using the backend
Tensor sum(Tensor t)
{
	size_t length = t.shape().back();
	auto vec = t.toHost<uint8_t>();
	std::vector<uint8_t> res(t.size()/length);
	for(size_t i=0;i<res.size();i++) {
		size_t sum = std::accumulate(vec.begin()+i*length, vec.begin()+(i+1)*length, 0);
		res[i] = (sum != 0);
	}
	return createTensor({(intmax_t)res.size()}, DType::Bool, res.data());
}

int main()
{
	size_t num_category = 3;
	size_t bits_per_category = 1;

	size_t sdr_size = bits_per_category*num_category;
	TemporalMemory tm({(intmax_t)sdr_size}, 2, 2);

	Tensor last_state;
	for(size_t i=0;i<40;i++) {
		size_t categoery = i%num_category;
		Tensor x = encoder::category(categoery, num_category, bits_per_category);
		auto [pred, active] = tm.compute(x, last_state);
		auto prediction = sum(pred);
		std::vector<size_t> pred_category = decoder::category(prediction, num_category);

		std::cout << "input, prediction of next = " << categoery << ", ";
		if(pred_category.size() == 0)
			std::cout << "None";
		for(auto v : pred_category)
			std::cout << v << ' ';
		std::cout << '\n';

		//Let the TM learn
		tm.learn(active, last_state);
		last_state = active;
	}
}
