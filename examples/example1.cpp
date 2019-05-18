#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
//#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/SpatialPooler.hpp>
#include <Etaler/Encoders/Scalar.hpp>
using namespace et;

#include <iostream>

int main()
{
	//Create a SP that takes in 128 input bits and generates 32 bit representation
	//setDefaultBackend(std::make_shared<OpenCLBackend>());
	SpatialPooler sp({128}, {32});

	//Encode the value 0.1 into a 32 bit SDR
	Tensor x = encoder::scalar(0.1, 0, 1, 128, 12);

	std::cout << sp.compute(x) << std::endl;

	auto state = sp.states();
	sp.loadState(state);

	std::cout << x.exp() << std::endl;
}
