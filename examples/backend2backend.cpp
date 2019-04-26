#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Encoders/Scalar.hpp>
using namespace et;

#include <iostream>

int main()
{
	auto gpu = std::make_shared<OpenCLBackend>();

	Tensor t = encoder::scalar(0.1, 0, 1, 32, 4);
	std::cout << "Tensor t on " + t.backend()->name() << std::endl;
	std::cout << "Content: " << t << std::endl << std::endl;

	std::cout << "Copy t to GPU" << std::endl << std::endl;
	Tensor q = t.to(gpu);

	std::cout << "Tensor q on " + q.backend()->name() << std::endl;
	std::cout << "Content: " << q << std::endl;
}
