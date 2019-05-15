#ifndef NUM_CELLS
	#error "NUM_CELLS is not defined"
#endif

#ifndef MAX_SYNAPSE_PER_CELL
	#error "MAX_SYNAPSE_PER_CELL is not defined"
#endif


kernel void decaySynapses(global int* restrict connections, global float* restrict permeances, float threshold)
{
	int global_size = get_global_size(0);
	int global_id = get_global_id(0);

	for(int i=global_id;i<NUM_CELLS;i+=global_size) {
		global int* synapses = connections+i*MAX_SYNAPSE_PER_CELL;
		global float* strengths = permeances+i*MAX_SYNAPSE_PER_CELL;

		int synapse_end = MAX_SYNAPSE_PER_CELL-1;
		for(;synapse_end!=0;synapse_end--) {
			if(synapses[synapse_end] != -1)
				break;
		}

		for(int j=0;j<=synapse_end;j++) {
			if(strengths[j] < threshold) {
				//swap a later synapse to the current one and disable the source one
				synapses[j] = synapses[synapse_end];
				synapses[synapse_end] = -1;

				strengths[j] = strengths[synapse_end];
				//strengths[synapse_end] = 0; //Don't care
				j -= 1;//Rerun the current connection
				synapse_end -= 1;
			}
		}
	}
}