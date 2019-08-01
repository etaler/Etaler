#ifndef INPUT_SIZE
	#error "INPUT_SIZE not defined"
#endif

#ifndef MAX_SYNAPSE_PER_CELL
	#error "MAX_SYNAPSE_PER_CELL not defined"
#endif

#ifndef PERM_TYPE
	#error "PERM_TYPE not defined"
#endif

#ifndef NO_UNUSED_SYNAPSE
	#define NO_UNUSED_SYNAPSE false
#endif

//global_size: Arbitrary
//local_size:  Arbitrary, but prefer multipel of CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
//INPUT_SIZE: The size of input SDR, must be smaller then CL_DEVICE_LOCAL_MEMORY_SIZE
//NO_UNUSED_SYNAPSE: If there are unised synapses. Useful for sparial pooler, accelerates ~30%
kernel void cellActivity(global bool* restrict x, global int* restrict synapses
	, global PERM_TYPE* restrict permeances, global int* restrict y
	, float connected_perm, int active_threshold, int output_size)
{
	int global_size = get_global_size(0);
	int global_id = get_global_id(0);

	int step = max(1, INPUT_SIZE/global_size);
	if(global_id < output_size) {
		int start = global_id*step;
		int end = (global_id+1)*step;

		for(int i=start;i<end;i++) {
			int sum = 0;
			for(int j=0;j<MAX_SYNAPSE_PER_CELL;j++) {
				int idx = i*MAX_SYNAPSE_PER_CELL+j;
				int target_cell = synapses[idx];

				if(!NO_UNUSED_SYNAPSE && target_cell == -1)
					break;

				// Accessing local memory is way faster then global. So test if the connected is on before
				// checking permeance
				if(x[target_cell] == 0)
					continue;

				float permeance = permeances[idx];

				//if(permeance >= connected_perm)
				//	sum += 1;
				//Same thing, but faster
				sum += (permeance >= connected_perm);
			}
			if(sum >= active_threshold)
				y[i] = sum;
			else
				y[i] = 0;
		}
	}
}
