#ifndef NUM_CELLS
	#error "NUM_CELLS is not defined"
#endif

#ifndef NUM_INPUT_BITS
	#error "NUM_INPUT_BITS is not defined"
#endif

#ifndef MAX_SYNAPSE_PER_CELL
	#error "MAX_SYNAPSE_PER_CELL is not defined"
#endif

//NOTE: The old version of this kernel might perform better with more cells

//global_size: Arbitrary
//local_size:  Arbitrary, but prefer multipel of CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
//NUM_CELLS: The number of cells in the layer
//NUM_INPUT_BITS: Number of bits the input SDR has
//MAX_SYNAPSE_PER_CELL: The max amount of connections a cell can have
//x: The input SDR **IN SPARSE FORMAT**
//aux: temporary buffer for storage, must be size of NUM_INPUT_BITS*global_size[0]
kernel void growSynapses(global int* restrict x, global bool* restrict y, global int* restrict connections
	, global float* restrict permeances, float initial_perm, int num_input_on_bits, global bool* restrict aux)
{
	int global_size = get_global_size(0);
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);
	int group_id = get_group_id(0);
	int group_size = get_num_groups(0);

	local int write_idx;

	for(int i=group_id;i<NUM_CELLS;i+=group_size) {
		if(y[i] == 0)
			continue;
		global int* restrict synapses = connections+i*MAX_SYNAPSE_PER_CELL;
		global float* restrict strengths = permeances+i*MAX_SYNAPSE_PER_CELL;
		global int const* restrict end = synapses+MAX_SYNAPSE_PER_CELL;
		global bool* restrict connection_list = aux+group_id*NUM_INPUT_BITS;

		if(*(end-1) != -1) //If the last slot is not empty, we are full and we don't need to dod anything
			continue;

		if(local_id == 0)
			write_idx = MAX_SYNAPSE_PER_CELL;
		barrier(CLK_LOCAL_MEM_FENCE);

		#pragma unroll 4
		for(int j=local_id;j<NUM_INPUT_BITS;j+=local_size)
			connection_list[j] = false;

		int local_min = MAX_SYNAPSE_PER_CELL;
		for(int j=local_id;j<MAX_SYNAPSE_PER_CELL;j+=local_size) {
			int idx = synapses[j];
			if(idx == -1) {
				atomic_min(&write_idx, j);
				break;
			}
			connection_list[idx] = true;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		for(int j=local_id;j<num_input_on_bits&&write_idx<MAX_SYNAPSE_PER_CELL;j+=local_size) {
			int idx = x[j];
			bool connected = connection_list[idx];

			if(connected == false) {
				int index = atomic_inc(&write_idx);
				if(index >= MAX_SYNAPSE_PER_CELL)
					break;
				synapses[index] = idx;
				strengths[index] = initial_perm;
			}
		}
	}
}