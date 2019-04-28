#ifndef NUM_CELLS
	#error "NUM_CELLS is not defined"
#endif

#ifndef NUM_INPUT_BITS
	#error "NUM_INPUT_BITS is not defined"
#endif

#ifndef MAX_SYNAPSE_PER_CELL
	#error "MAX_SYNAPSE_PER_CELL is not defined"
#endif

kernel void growSynapses(global bool* restrict x, global bool* restrict y, global int* restrict connections
	, global float* restrict permeances, float initial_perm)
{
	int global_size = get_global_size(0);
	int global_id = get_global_id(0);

	for(int i=global_id;i<NUM_CELLS;i+=global_size) {
		if(y[i] == 0)
			continue;
		global int* synapses = connections+i*MAX_SYNAPSE_PER_CELL;
		global float* strengths = permeances+i*MAX_SYNAPSE_PER_CELL;
		global int* end = synapses+MAX_SYNAPSE_PER_CELL;

		if(*(end-1) != -1)
			continue;

		global int* synapse_end = synapses;
		for(;synapse_end!=end;synapse_end++) {
			if(*synapse_end == -1)
				break;
		}

		int avliable_space = end - synapse_end;
		int used_space = synapse_end - synapses;

		int write_idx = synapse_end - synapses;
		int read_idx = 0;

		for(int j=0;avliable_space!=0 && j<NUM_INPUT_BITS;j++) {
			if(x[j] == false)
				continue;
			bool connected = false;
			for(;read_idx<used_space;read_idx++) {
				if(synapses[read_idx] == j) {
					connected = true;
					break;
				}
				if(synapses[read_idx] > j)
					break;
			}

			if(connected == false) {
				synapses[write_idx] = j;
				strengths[write_idx] = initial_perm;
				avliable_space--;
				write_idx++;
			}
		}
	}
}