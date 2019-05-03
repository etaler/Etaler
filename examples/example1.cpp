#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/SpatialPooler.hpp>
#include <Etaler/Encoders/Scalar.hpp>
using namespace et;

#include <iostream>

int main()
{
	//Create a SP that takes in 128 input bits and generates 32 bit representation
	/*SpatialPooler sp({128}, {32});

	//Encode the value 0.1 into a 32 bit SDR
	Tensor x = encoder::scalar(0.1, 0, 1, 128, 12);

	std::cout << sp.compute(x) << std::endl;

	auto state = sp.states();
	sp.loadState(state);*/

	auto backend = std::make_shared<OpenCLBackend>();
	setDefaultBackend(backend);

	std::vector<int> data(16);
	for(size_t i=0;i<data.size();i++)
		data[i] = i;
	Tensor t = createTensor({4,4}, DType::Int32, data.data());

	Tensor q = t.view({range(2),range(2)});
	std::cout << q.size() << std::endl;
	std::cout << attempt_realize(q) << std::endl;
}
