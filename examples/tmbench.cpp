#include <Etaler/Etaler.hpp>
#include <Etaler/Backends/CPUBackend.hpp>
#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Encoders/Scalar.hpp>
using namespace et;

#include <vector>
#include <chrono>
#include <random>

float benchmarkTemporalMemory(const std::vector<Tensor>& x, size_t num_epoch)
{
	float time_used = 0;
	for(size_t i=0;i<num_epoch;i++) {

		TemporalMemory tm(x[0].shape(), 16, 1024);

		//Force OpenCL kernel precompilcation
		Tensor t = zeros(x[0].shape() + 16, DType::Bool);
		tm.compute(x[0], t);
		tm.learn(t, t);

		Tensor last_state = zeros(t.shape(), DType::Bool);

		auto t0 = std::chrono::high_resolution_clock::now();
		for(const auto& d : x) {
			auto [pred, active] = tm.compute(d, last_state);
			tm.learn(active, last_state);
			last_state = active;
		}
		defaultBackend()->sync();

		auto t1 = std::chrono::high_resolution_clock::now();

		time_used += std::chrono::duration_cast<std::chrono::duration<float>>(t1-t0).count();
	}

	return time_used/num_epoch;
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
	std::shared_ptr<Backend> backend = std::make_shared<OpenCLBackend>();
	setDefaultBackend(backend);

	std::cout << "Benchmarking TemporalMemory algorithm on backend: " << defaultBackend()->name() <<" \n\n";

	std::vector<Tensor> input_data;
	std::vector<size_t> input_size = {64, 128, 512, 1024, 8192};
	size_t num_data = 1000;

	for(auto input_len : input_size) {
		auto input_data = generateRandomData(input_len, num_data);

		float t = benchmarkTemporalMemory(input_data, 1);
		std::cout << input_len << " bits per SDR, " << t/num_data*1000 << "ms per forward" << std::endl;
		//std::cout << input_len << "," << t/num_data*1000 << std::endl;
	}
}
