#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Encoders/Category.hpp>
using namespace et;

#include <iostream>

int main()
{
	TemporalMemory tm({2}, 2, 2);

	Tensor last_state;
	for(size_t i=0;i<6;i++) {
		Tensor x = encoder::category(i%2, 2, 1);
		auto [pred, active] = tm.compute(x, last_state);
		std::cout << pred << std::endl;
		tm.learn(active, last_state);
		last_state = active;
	}
}
