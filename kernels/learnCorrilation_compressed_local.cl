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
//INPUT_SIZE: The size of input SDR, must be smaller then CL_DEVICE_LOCAL_MEMORY_SIZE*8-8
//NO_UNUSED_SYNAPSE: If there are unised synapses. Useful for sparial pooler, accelerates ~30%
kernel void learnCorrilation(global bool* restrict x, global bool* restrict y
	, global int* restrict synapses, global PERM_TYPE* restrict permeances
	, float permeance_inc, float permeance_dec)
{
	local char xl[INPUT_SIZE/8+1];
	size_t id = get_local_id(0);
	size_t size = get_local_size(0);

	for(int i=id;i<INPUT_SIZE/8;i+=size) {
		unsigned char res = 0;
		#pragma unroll
		for(int j=0;j<8;j++)
			res |= x[i*8+j] << j;
		xl[i] = res;
	}

	if(id == 0 && INPUT_SIZE%8 != 0) {
		int i=INPUT_SIZE/8;
		unsigned char res = 0;
		for(int j=0;j<8 && i*8+j < INPUT_SIZE;j++)
			res |= x[i*8+j] << j;
		xl[i] = res;
	}

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
			bool x = (xl[target_cell/8] & (1 << target_cell%8)) != 0;
			if(x == true)
				permeance += permeance_inc;
			else
				permeance -= permeance_dec;

			permeances[idx] = clamp(permeance, 0.f, 1.f);
		}
	}
}