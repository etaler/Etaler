#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
//#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Encoders/Category.hpp>
using namespace et;

#include <iostream>
#include <numeric>

int main()
{
	//auto backend = std::make_shared<et::OpenCLBackend>();
	//et::setDefaultBackend(backend.get());
	size_t num_category = 3;
	size_t bits_per_category = 5;

	size_t sdr_size = bits_per_category*num_category;
	TemporalMemory tm({(intmax_t)sdr_size}, 2);

	Tensor last_state;
	for(size_t i=0;i<40;i++) {
		size_t categoery = i%num_category;
		Tensor x = encoder::category(categoery, num_category, bits_per_category);
		auto [pred, active] = tm.compute(x, last_state);
		auto prediction = sum(pred, 1).cast(DType::Bool);
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
