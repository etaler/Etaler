#ifndef InType
        #error InType not defined
#endif

#ifndef OutType
        #error OutType not defined
#endif

#ifndef IntermidType
        #error IntermidType not defined
#endif

#ifdef IntermidIsHalf
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Just a sane number for GPUs since quering the number of compute units in a CU in OpenCL
// is quite ganky. TODO: The number needs to be changed for a FPGA or a VLIW processor (
// anything that's not SIMT)
#define WORKITEM_PER_CU 64

//InType: Input Data type
//OutType: Output Data type
//in_size: number of elements of the input
//chunk_size: for each chunk_size elements, produce 1 sum
//local_size: must equal to WORKITEM_PER_CU
//group_size: must equal to in_size/chunk_size
kernel void sum(global InType* restrict x, global OutType* restrict y, int in_size, int chunk_size)
{
        local IntermidType local_sum[WORKITEM_PER_CU];
        int group_id = get_group_id(0);
        int group_size = get_num_groups(0);
        int local_size = get_local_size(0);
        int local_id = get_local_id(0);
        IntermidType private_sum = 0;
        int start = chunk_size*group_id;
        for(int i=start+local_id;i<start+chunk_size; i+=local_size)
                private_sum += x[i];
        local_sum[local_id] = private_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        // reduce the indivisually computed local result into a final sum
        if(local_id == 0) {
                IntermidType s = 0;
                for(int i=0;i<local_size;i++)
                        s += local_sum[i];
                y[group_id] = s;
        }
}