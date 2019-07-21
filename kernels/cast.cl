#ifndef InType
	#error InType not defined
#endif

#ifndef OutType
	#error OutType not defined
#endif

#ifdef HalfSupport
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

//InType: Input Type
//OutType: OutputType
//global_size: arbitrary
kernel void cast(global InType* restrict x, global OutType* restrict y, int problem_size)
{
	int id = get_global_id(0);
	int size = get_global_size(0);
	for(int i=id;i<problem_size;i+=size)
		y[i] = (OutType)x[i];
}