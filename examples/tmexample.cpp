#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
//#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Encoders/Category.hpp>
using namespace et;

#include <iostream>
#include <numeric>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

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
	size_t num_category = 3;
	size_t bits_per_category = 5;
	intmax_t cells_per_column = 2;

	intmax_t sdr_size = bits_per_category*num_category;
	TemporalMemory tm({(intmax_t)sdr_size}, cells_per_column);

	Tensor last_state = zeros({sdr_size, cells_per_column}, DType::Bool);
	for(size_t i=0;i<40;i++) {
		size_t categoery = i%num_category;
		Tensor x = encoder::category(categoery, num_category, bits_per_category);

		auto [pred, active] = tm.compute(x, last_state);

		tm.learn(active, last_state); //Let the TM learn
		last_state = active;

		//Display results
		auto prediction = sum(pred, 1, DType::Bool);
		std::vector<size_t> pred_category = decoder::category(prediction, num_category);

		std::cout << "input, prediction of next = " << categoery
			<< ", " << to_string(pred_category);
		std::cout << '\n';
	
		if (i==0)
		{
			std::vector<std::vector<double>> x1, y1, z1;
    		for (double i = -5; i <= 5;  i += 0.25) {
				std::vector<double> x_row, y_row, z_row;
				for (double j = -5; j <= 5; j += 0.25) {
					x_row.push_back(i);
					y_row.push_back(j);
					z_row.push_back(::std::sin(::std::hypot(i, j)));
				}
				x1.push_back(x_row);
				y1.push_back(y_row);
				z1.push_back(z_row);
			}		

			plt::plot_scatter(x1,y1,z1);;
			plt::show();
		}
		
	}
}
