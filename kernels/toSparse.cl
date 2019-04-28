//Not used

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
kernel void onBits(global bool* restrict x, global int* restrict y)
{
	local int count;
	int id = get_local_id(0);
	int size = get_local_size(0);
	if(id == 0)
		count = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int i=id;i<INPUT_SIZE;i+=size) {
		if(x[i] == true)
			atomic_inc(&count);
	}
	*y = count;
}


kernel void toSparse(global bool* restrict x, global int* restrict y)
{
	local int count;
	int id = get_local_id(0);
	int size = get_local_size(0);
	if(id == 0)
		count = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int i=id;i<INPUT_SIZE;i+=size) {
		if(x[i] == true)
			y[atomic_inc(&count)] = i;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	y[count] = -1;
}
