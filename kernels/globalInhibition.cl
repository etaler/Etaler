#ifndef INPUT_SIZE
	#error "INPUT_SIZE not defined"
#endif

#ifndef MAX_INPUT_VALUE
	#error "MAX_INPUT_VALUE not defined"
#endif

#if MAX_INPUT_VALUE < 1
	#error "MAX_INPUT_VALUE >= 0: Assertion failed."
#endif

//This method runs in O(n) time, but the max value it can process is determined by the amount of local memory
//global_size: Arbitrary
//local_size:  Arbitrary, but prefer multipel of CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
//INPUT_SIZE: The size of input SDR, must be smaller then CL_DEVICE_LOCAL_MEMORY_SIZE
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
kernel void fastTopK(global int* restrict x, global int* restrict result, int k)
{
	local unsigned int res[MAX_INPUT_VALUE];
	int size = get_local_size(0);
	int id = get_local_id(0);

	for(int i=id;i<MAX_INPUT_VALUE;i+=size)
		res[i] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for(int i=id;i<INPUT_SIZE;i+=size) {
		int v = x[i];
		if(v < MAX_INPUT_VALUE)
			atomic_inc(res+v);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(id == 0) {
		int n = INPUT_SIZE-k;
		int occur_sum = res[0];
		int max_val = 0;
		bool solved = 0;
		#pragma unroll 4
		for(int i=1;i<MAX_INPUT_VALUE;i++) {
			int occur = res[i];
			occur_sum += occur;
			if(occur_sum > n) {
				*result = i;
				solved = true;
				break;
			}

			if(occur > 0)
				max_val = i;
		}

		if(solved == false)
			*result = max_val;
	}
}

kernel void threshold(global int* restrict x, global bool* restrict y, global int* restrict threshold)
{
	int id = get_global_id(0);
	int size = get_global_size(0);
	int thr = *threshold;
	for(int i=id;i<INPUT_SIZE;i+=size) {
		int v = x[i];
		y[i] = (v >= thr ? 1 : 0);
	}
}
