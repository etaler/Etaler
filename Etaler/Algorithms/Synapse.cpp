#include "Synapse.hpp"
#include "Etaler/Core/Random.hpp"

using namespace et;


template <typename T=size_t>
static inline std::vector<T> vector_range(T start, T end)
{
	std::vector<T> v(end-start);
	for(T i=0;i<end-start;i++)
		v[i] = i;
	return v;
}

std::pair<Tensor, Tensor> et::F::gusianRandomSynapse(const Shape& input_shape, const Shape& output_shape, float potential_pool_pct
	, float mean, float stddev , size_t seed, Backend* backend)
{
	if(potential_pool_pct > 1 || potential_pool_pct <= 0)
		throw EtError("potential_pool_pct must be in range of (0, 1], but get" + std::to_string(potential_pool_pct));
        if(mean > 1 || mean < 0)
                throw EtError("mean must be in range [0, 1]. But get " + std::to_string(mean));
        if(stddev <= 0)
                throw EtError("stddev must be larger than 0 " + std::to_string(stddev));

	size_t input_cell_num = input_shape.volume();
	size_t potential_pool_size = std::max(size_t(input_cell_num*potential_pool_pct), size_t{1});
        Shape synapse_shape = output_shape + potential_pool_size;

	// Initalize potential pool
        // First generate the synapses
	pcg64 rng(seed);
	std::vector<int32_t> connections(output_shape.volume()*potential_pool_size, -1);

        std::vector<int32_t> all_input_cell = vector_range<int32_t>(0, input_cell_num);
	for(intmax_t i=0;i<output_shape.volume();i++) {
		std::shuffle(all_input_cell.begin(), all_input_cell.end(), rng);
                // Synapses might need to be sorted for some backends
		std::sort(all_input_cell.begin(), all_input_cell.begin()+potential_pool_size);
                std::copy(all_input_cell.begin(), all_input_cell.begin()+potential_pool_size
                        , connections.begin()+i*potential_pool_size);
	}

        // Generate all the permanences
        std::normal_distribution<float> dist(mean, stddev);
        std::vector<float> permanences(output_shape.volume()*potential_pool_size);
        std::generate(permanences.begin(), permanences.end(), [&rng, &dist](){return std::clamp(dist(rng), 0.f, 1.f);});

        // Turn the generaed data into tnesors
        et_assert(synapse_shape.volume() == intmax_t(connections.size()));
        et_assert(synapse_shape.volume() == intmax_t(permanences.size()));
	Tensor conn = Tensor(synapse_shape, connections.data(), backend);
	Tensor perm = Tensor(synapse_shape, permanences.data(), backend);
        return {conn, perm};
}

std::pair<Tensor, Tensor> et::F::gusianRandomSynapseND(const Shape& input_shape, size_t kernel_size, size_t stride, float potential_pool_pct
	, float mean, float stddev, size_t seed, Backend* backend)
{
        if(potential_pool_pct > 1 || potential_pool_pct <= 0)
		throw EtError("potential_pool_pct must be in range of (0, 1], but get" + std::to_string(potential_pool_pct));
        if(mean > 1 || mean < 0)
                throw EtError("mean must be in range [0, 1]. But get " + std::to_string(mean));
        if(stddev <= 0)
                throw EtError("stddev must be larger than 0 " + std::to_string(stddev));

	for(size_t i=0;i<input_shape.size();i++)
		et_assert(input_shape[i] >= (intmax_t)kernel_size, "dimension must be larger than kernel size");

	Shape output_shape(input_shape.size());
	for(size_t i=0;i<input_shape.size();i++)
		output_shape[i] = (input_shape[i]-kernel_size)/stride+1;

	size_t input_cell_num = input_shape.volume();
	size_t potential_pool_size = std::max((size_t)(std::pow(kernel_size, input_shape.size())*potential_pool_pct), size_t{1});
        Shape synapse_shape = output_shape + potential_pool_size;

	Tensor indices = Tensor(vector_range<int>(0, input_cell_num), backend).reshape(input_shape);

        // Generating the synapses for ND synapses are more complicated
        pcg64 rng(seed);
	Tensor conn = Tensor(synapse_shape, DType::Int32, backend);
	for(size_t i=0;i<(size_t)output_shape.volume();i++) {
		IndexList write_loc;
		Shape loc = foldIndex(i, output_shape);
		for(size_t j=0;j<output_shape.size();j++)
			write_loc.push_back(loc[j]);

		IndexList read_loc(loc.size());
		for(size_t j=0;j<loc.size();j++) {
			intmax_t pos = loc[j]*stride;
			read_loc[j] = range(pos, pos+kernel_size);
		}

		std::vector<int> conns = indices.view(read_loc).toHost<int>();
		assert(conns.size() >= potential_pool_size);
		std::shuffle(conns.begin(), conns.end(), rng);
		std::sort(conns.begin(), conns.begin()+potential_pool_size);
		conn.view(write_loc) = Tensor({(intmax_t)potential_pool_size}, conns.data());
	}

        // Generate all the permanences
        std::normal_distribution<float> dist(mean, stddev);
        std::vector<float> permanences(output_shape.volume()*potential_pool_size);
        std::generate(permanences.begin(), permanences.end(), [&rng, &dist](){return std::clamp(dist(rng), 0.f, 1.f);});
        Tensor perm = Tensor(synapse_shape, permanences.data(), backend);

        return {conn, perm};
}