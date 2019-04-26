#pragma once

#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <random>

#include <Etaler/Core/Random.hpp>
#include <Etaler/Core/Tensor.hpp>
#include <Etaler/Core/Backend.hpp>

namespace et
{

namespace encoder
{

//This implementation is slower the one in tiny-htm. But is more accurate
static std::vector<uint8_t> gcm1d(float v, float scale, size_t active_cells, size_t num_cells)
{
	std::vector<uint8_t> res(num_cells);
	float loc = v*scale;

	std::vector<std::pair<size_t, float>> cell_distances;
	cell_distances.reserve(res.size());

	//Use a faster algoithm for the special case of active_cells == 1
	if(active_cells != 1) {
		for(size_t i=0;i<res.size();i++) {
			float dist = std::fmod(std::abs(loc-i), num_cells);
			cell_distances.push_back({i, dist});
		}

		std::nth_element(cell_distances.begin(), cell_distances.begin()+active_cells, cell_distances.end(),[](auto a, auto b){return a.second < b.second;});
		for(size_t i=0;i<active_cells;i++)
			res[cell_distances[i].first] = 1;
	}
	else {
		float dist = std::fmod(loc, num_cells);
		if(dist < 0)
			dist += num_cells;
		size_t index = (size_t)dist + (std::fmod(dist, 1) > 0.5f ? 1 : 0);
		res[index] = 1;
	}
	return res;
}

static Tensor gridCell1d(float v, size_t num_gcm=16, size_t active_cells_per_gcm=1, size_t length_per_gcm=16
	, std::array<float, 2> scale_range={0.1f, 4.f}, size_t seed=42, Backend* backend=defaultBackend())
{
	std::vector<uint8_t> encoding(num_gcm*length_per_gcm);
	pcg32 rng(seed);
	std::uniform_real_distribution<float> scale_dist(scale_range[0], scale_range[1]);

	for(size_t i=0;i<num_gcm;i++) {
		auto gcm_res = gcm1d(v, scale_dist(rng), active_cells_per_gcm, length_per_gcm);
		std::copy(gcm_res.begin(), gcm_res.end(), encoding.begin()+i*length_per_gcm);
	}

	return backend->createTensor({(intmax_t)encoding.size()}, DType::Bool, encoding.data());
}

}

}
