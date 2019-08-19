#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
//#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Encoders/Category.hpp>
using namespace et;

#include <iostream>
#include <numeric>


inline std::string to_string(std::vector<size_t> v)
{
	if(v.size() == 0)
		return "None";

	std::string res = "";
	for(size_t i=0;i<v.size();i++)
		res += std::to_string(v[i]) + std::string(i==v.size()-1 ? "" : " ");
	return res;
}

int main()
{
	//auto backend = std::make_shared<et::OpenCLBackend>();
	//et::setDefaultBackend(backend.get());

	//Parameters for the example
	size_t num_category = 3;
	size_t bits_per_category = 5;
	intmax_t cells_per_column = 2;

	//Initalize the Temporal Memory layer
	intmax_t sdr_size = bits_per_category*num_category;
	TemporalMemory tm({(intmax_t)sdr_size}, cells_per_column);

	//Intermid HTM states
	Tensor last_state = zeros({sdr_size, cells_per_column}, DType::Bool);
	Tensor last_pred = zeros({sdr_size, cells_per_column}, DType::Bool);

	//Iterate a few times
	for(size_t i=0;i<40;i++) {
		size_t categoery = i%num_category;

		//Create SDR base on which iteration we are in
		Tensor x = encoder::category(categoery, num_category, bits_per_category);

		//Ask the TM to make predictions
		auto [pred, active] = tm.compute(x, last_pred);

		//Let the TM learn
		tm.learn(active, last_state);
		last_state = active;
		last_pred = pred;

		//Display results
		auto prediction = sum(pred, 1, DType::Bool); // extract what the TM is predicting
		auto pred_category = decoder::category(prediction, num_category);

		std::cout << "input, prediction of next = " << categoery
			<< ", " << to_string(pred_category);
		std::cout << '\n';


	}
}