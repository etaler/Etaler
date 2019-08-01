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

int up_round(int v, int mul)
{
	return (v/mul + v%mul!=0)*mul;
}

//global_size: Arbitrary
//local_size:  Arbitrary, but prefer multipel of CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
//INPUT_SIZE: The size of input SDR, must be smaller then CL_DEVICE_LOCAL_MEMORY_SIZE*8-8
//NO_UNUSED_SYNAPSE: If there are unised synapses. Useful for sparial pooler, accelerates ~30%
kernel void cellActivity(global bool* restrict x, global int* restrict synapses
	, global PERM_TYPE* restrict permeances, global int* restrict y
	, float connected_perm, int active_threshold, int output_size)
{
	//Load input state into local memory for faster access
	local unsigned char xl[INPUT_SIZE/8+1];
	int id = get_local_id(0);
	int size = get_local_size(0);

	for(int i=id;i<INPUT_SIZE/8;i+=size) {
		unsigned char res = 0;
		#pragma unroll
		for(int j=0;j<8;j++)
			res |= x[i*8+j] << j;
		xl[i] = res;
	}

	if(id == 0) {
		int i = (INPUT_SIZE - INPUT_SIZE%8)/8;
		unsigned char res = 0;
		#pragma unroll
		for(int j=0;j<8 && i+j < INPUT_SIZE;j++)
			res |= x[i*8+j] << j;
		xl[i] = res;
	}

	//Wait for all Work Item copying the data
	barrier(CLK_LOCAL_MEM_FENCE);

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

				float permeance = permeances[idx];
				bool x = (xl[target_cell/8] & (1 << target_cell%8)) != 0;
				if(x == 0)
					continue;

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
