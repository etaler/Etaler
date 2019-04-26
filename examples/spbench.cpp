#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
//#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/SpatialPooler.hpp>
#include <Etaler/Encoders/Scalar.hpp>
using namespace et;

#include <vector>
#include <chrono>

float benchmarkSpatialPooler(const Shape& out_shape, const std::vector<Tensor>& x, size_t num_epoch)
{
	SpatialPooler sp(x[0].shape(), out_shape);

	//To make the OpenCL backen ptr-compile the kernels
	sp.compute(x[0]);

	auto t0 = std::chrono::high_resolution_clock::now();
	for(size_t i=0;i<num_epoch;i++) {
		for(const auto& d : x)
			sp.compute(d);
	}
	defaultBackend()->sync();
	auto t1 = std::chrono::high_resolution_clock::now();

	return std::chrono::duration_cast<std::chrono::duration<float>>(t1-t0).count()/num_epoch;
}

std::vector<Tensor> generateRandomData(size_t input_length, size_t num_data)
{
	std::vector<Tensor> res(num_data);
	static std::mt19937 rng;
	std::uniform_real_distribution<float> dist(0, 1);

	for(size_t i=0;i<num_data;i++)
		res[i] = encoder::scalar(dist(rng), 0, 1, input_length, input_length*0.15);
	return res;
}

int main()
{
	std::shared_ptr<Backend> backend = std::make_shared<CPUBackend>();
	setDefaultBackend(backend);

	std::cout << "Benchmarking SpatialPooler algorithm on backend: " << backend->name() <<" \n\n";

	std::vector<Tensor> input_data;
	std::vector<size_t> input_size = {64, 128, 256, 512, 1024, 2048, 9000};
	size_t num_data = 1000;

	for(auto input_len : input_size) {
		auto input_data = generateRandomData(input_len, num_data);

		float t = benchmarkSpatialPooler({(int)input_len}, input_data, 1);
		std::cout << input_len << " bits per SDR, " << t/num_data*1000 << "ms per forward" << std::endl;
		//std::cout << input_len << "," << t/num_data*1000 << std::endl;
	}
}
