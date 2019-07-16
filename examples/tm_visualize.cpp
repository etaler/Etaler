#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
//#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Encoders/Category.hpp>
using namespace et;

#include <iostream>
#include <numeric>
#include <unistd.h>
#include <Visualizer.hpp>

inline std::string to_string(std::vector<size_t> v)
{
	if(v.size() == 0)
		return "None";

	std::string res = "";
	for(size_t i=0;i<v.size();i++)
		res += std::to_string(v[i]) + std::string(i==v.size()-1 ? "" : " ");
	return res;
}


int main(int argc, char **argv)
{
	char the_path[256];

    getcwd(the_path, 255);
    printf("Executable is run from %s - please make sure shaders resources are on that path...\n", the_path);	//auto backend = std::make_shared<et::OpenCLBackend>();
	//et::setDefaultBackend(backend.get());
	size_t num_category = 3;
	size_t bits_per_category = 5;
	intmax_t cells_per_column = 2;
	
	intmax_t sdr_size = bits_per_category*num_category;
	TemporalMemory tm({(intmax_t)sdr_size}, cells_per_column);

	Tensor last_state = zeros({sdr_size, cells_per_column}, DType::Bool);

	Visualizer * vis;
	vis = new Visualizer(cells_per_column, bits_per_category*num_category);

	for(size_t i=0;i<40;i++) {
		size_t categoery = i%num_category;
		Tensor x = encoder::category(categoery, num_category, bits_per_category);

		auto [pred, active] = tm.compute(x, last_state);

		bool * active_buff = (bool*)active.data();
		vis->UpdateLayer(0, active_buff);

		//std::cout << active << std::endl;
		usleep(1000000);

		tm.learn(active, last_state); //Let the TM learn
		last_state = active;

		//Display results
		auto prediction = sum(pred, 1, DType::Bool);
		std::vector<size_t> pred_category = decoder::category(prediction, num_category);

		std::cout << "input, prediction of next = " << categoery
			<< ", " << to_string(pred_category);
		std::cout << '\n';
	}

}
