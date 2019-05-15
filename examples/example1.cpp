#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/SpatialPooler.hpp>
#include <Etaler/Encoders/Scalar.hpp>
using namespace et;

#include <iostream>

int main()
{
	/*
	//Create a SP that takes in 128 input bits and generates 32 bit representation
	setDefaultBackend(std::make_shared<OpenCLBackend>());
	SpatialPooler sp({128}, {32});

	//Encode the value 0.1 into a 32 bit SDR
	Tensor x = encoder::scalar(0.1, 0, 1, 128, 12);

	std::cout << sp.compute(x) << std::endl;

	auto state = sp.states();
	sp.loadState(state);*/

	setDefaultBackend(std::make_shared<OpenCLBackend>());

	int a[] = {0,1,0,1};
	Tensor c({2,2}, a);

	float b[] = {0.1, 0.7, 0.5, 0.01};
	Tensor p({2,2}, b);

	std::cout << c << std::endl;
	std::cout << p << std::endl;

	defaultBackend()->decaySynapses(c, p, 0.2);

	std::cout << c << std::endl;
	std::cout << p << std::endl;
}
