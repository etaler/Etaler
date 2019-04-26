#pragma once

#include <vector>
#include <array>
#include <random>
#include <cmath>

#include <Etaler/Core/Random.hpp>
#include <Etaler/Core/Tensor.hpp>
#include <Etaler/Core/Backend.hpp>

namespace et
{

namespace encoder
{

static std::array<float, 4> inv2x2(std::array<float, 4> m)
{
	float det = m[0]*m[3] - m[1]*m[2];
	float f = 1/det;

	return {f*m[3], -f*m[1], -f*m[2], f*m[0]};
}

static std::array<float, 2> mvmul2(std::array<float, 4> m, std::array<float, 2> v, float f)
{
	return {(v[0]*m[0]+v[1]*m[1])*f, (v[0]*m[2]+v[1]*m[3])*f};
}

//This implementation is slower the one in tiny-htm. But is more accurate
static std::vector<uint8_t> gcm2d(std::array<float, 2> p, float scale, float theta, size_t active_cells, std::array<size_t, 2> axis_length)
{
	std::vector<uint8_t> res(axis_length[0]*axis_length[1]);
	std::array<float, 4> mat = inv2x2({cosf(theta), -sinf(theta)
					,sinf(theta), cosf(theta)});

	std::array<float, 2> np = mvmul2(mat, p, scale);
	std::vector<std::pair<size_t, float>> cell_distances;
	cell_distances.reserve(res.size());

	float px = std::fmod(np[0], axis_length[0]) + (np[0] < 0 ? axis_length[0] : 0);
	float py = std::fmod(np[1], axis_length[1]) + (np[1] < 0 ? axis_length[1] : 0);

	for(size_t i=0;i<res.size();i++) {

		float x = i%axis_length[0] + 0.5;
		float y = axis_length[1] - (i/axis_length[0] + 0.5); //Flip axis for math conventions

		float dx = px - x;
		float dy = py - y;

		float dist = std::sqrt(dx*dx + dy*dy);

		cell_distances.push_back({i, dist});
	}

	std::nth_element(cell_distances.begin(), cell_distances.begin()+active_cells, cell_distances.end(),[](auto a, auto b){return a.second < b.second;});
	for(size_t i=0;i<active_cells;i++)
		res[cell_distances[i].first] = 1;
	return res;
}

static Tensor gridCell2d(std::array<float, 2> p, size_t num_gcm=16, size_t active_cells_per_gcm=1, std::array<size_t, 2> gcm_axis_length={4,4}
	, std::array<float, 2> scale_range={0.3f, 1.f}, size_t seed=42, Backend* backend=defaultBackend())
{
	const float pi = 3.14159265358979323846;
	size_t gcm_size = gcm_axis_length[0]*gcm_axis_length[1];
	std::vector<uint8_t> encoding(num_gcm*gcm_size);
	pcg32 rng(seed);
	std::uniform_real_distribution<float> rotation_dist(0, 2.f*pi);
	std::uniform_real_distribution<float> scale_dist(scale_range[0], scale_range[1]);

	for(size_t i=0;i<num_gcm;i++) {
		auto gcm_res = gcm2d(p, scale_dist(rng), rotation_dist(rng), active_cells_per_gcm, gcm_axis_length);
		std::copy(gcm_res.begin(), gcm_res.end(), encoding.begin()+i*gcm_size);
	}

	return backend->createTensor({(intmax_t)encoding.size()}, DType::Bool, encoding.data());
}

}

}
