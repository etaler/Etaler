#ifndef INPUT_SIZE
	#error "INPUT_SIZE not defined"
#endif

#ifndef MAX_SYNAPSE_PER_CELL
	#error "MAX_SYNAPSE_PER_CELL not defined"
#endif

#ifndef OUTPUT_SIZE
	#error "OUTPUT_SIZE not defined"
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
kernel void learnCorrilation(global bool* restrict x, global bool* restrict y
	, global int* restrict synapses, global PERM_TYPE* restrict permeances
	, float permeance_inc, float permeance_dec)
{
	local char xl[INPUT_SIZE];
	size_t id = get_local_id(0);
	size_t size = get_local_size(0);
	for(size_t i=id;i<INPUT_SIZE;i+=size)
		xl[i] = x[i];

	barrier(CLK_LOCAL_MEM_FENCE);

	int global_size = get_global_size(0);
	int global_id = get_global_id(0);
	for(int i=global_id;i<OUTPUT_SIZE;i+=global_size) {
		if(y[i] == false)
			continue;

		for(int j=0;j<MAX_SYNAPSE_PER_CELL;j++) {
			int idx = i*MAX_SYNAPSE_PER_CELL+j;
			int target_cell = synapses[idx];

			if(!NO_UNUSED_SYNAPSE && target_cell == -1)
				break;

			float permeance = permeances[idx];
			if(xl[target_cell] == true)
				permeance += permeance_inc;
			else
				permeance -= permeance_dec;

			permeances[idx] = max(min(permeance, 1.f), 0.f);
		}
	}
}