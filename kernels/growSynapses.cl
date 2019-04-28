#ifndef NUM_CELLS
	#error "NUM_CELLS is not defined"
#endif

#ifndef NUM_INPUT_BITS
	#error "NUM_INPUT_BITS is not defined"
#endif

#ifndef MAX_SYNAPSE_PER_CELL
	#error "MAX_SYNAPSE_PER_CELL is not defined"
#endif
//global_size: Arbitrary
//local_size:  Arbitrary, but prefer multipel of CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
//NUM_CELLS: The number of cells in the layer
//NUM_INPUT_BITS: Number of bits the input SDR has
//MAX_SYNAPSE_PER_CELL: The max amount of connections a cell can have
//x: The input SDR **IN SPARSE FORMAT**
//aux: temporary buffer for storage, must be size of NUM_INPUT_BITS*global_size[0]
kernel void growSynapses(constant int* restrict x, global bool* restrict y, global int* restrict connections
	, global float* restrict permeances, float initial_perm, global bool* restrict aux)
{
	int global_size = get_global_size(0);
	int global_id = get_global_id(0);

	for(int i=global_id;i<NUM_CELLS;i+=global_size) {
		if(y[i] == 0)
			continue;
		global int* restrict synapses = connections+i*MAX_SYNAPSE_PER_CELL;
		global float* restrict strengths = permeances+i*MAX_SYNAPSE_PER_CELL;
		global int const* restrict end = synapses+MAX_SYNAPSE_PER_CELL;
		global bool* restrict connection_list = aux+global_id*NUM_INPUT_BITS;

		if(*(end-1) != -1) //If the last slot is not empty, we are full and we don't need to dod anything
			continue;

		#pragma unroll 4
		for(int j=0;j<NUM_INPUT_BITS;j++)
			connection_list[j] = false;

		int synapse_end = MAX_SYNAPSE_PER_CELL;
		for(int j=0;j<MAX_SYNAPSE_PER_CELL;j++) {
			int idx = synapses[j];
			if(idx == -1) {
				synapse_end = j;
				break;
			}
			connection_list[idx] = true;
		}

		int write_idx = synapse_end;


		for(int j=0;write_idx!=MAX_SYNAPSE_PER_CELL && x[j]!=-1;j++) {
			int idx = x[j];
			bool connected = connection_list[idx];

			if(connected == false) {
				synapses[write_idx] = idx;
				strengths[write_idx] = initial_perm;
				write_idx++;
			}
		}
	}
}