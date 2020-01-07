#ifndef INPUT_TYPE
        #error INPUT_TYPE not defined
#endif

#ifndef OUTPUT_TYPE
        #error OUTPUT_TYPE not defined
#endif

#define OPENCL_TENSOR_MAX_DIMS 32
typedef struct __attribute__ ((packed)) _View
{
        int stride[OPENCL_TENSOR_MAX_DIMS];
        int shape_stride[OPENCL_TENSOR_MAX_DIMS];
        int offset;
        int dims;
} View;

int offset_from_index(View view, int index)
{
	int curr_idx = index;
	int sum = 0;
	for(int i=0;i<view.dims;i++) {
		int s = view.shape_stride[i];
		int ndpos = curr_idx / s;
		sum += ndpos * view.stride[i];
		curr_idx %= s;
	}
	return sum + view.offset;
}

kernel void copy(global OUTPUT_TYPE* out, global INPUT_TYPE* in, View output_view,  View input_view, int problem_size)
{
        int global_id = get_global_id(0);
        int global_size = get_global_size(0);

        View in_view = input_view;
        View out_view = output_view;

        for(int i=global_id;i<problem_size;i+=global_size) {
                int out_idx = offset_from_index(out_view, i);
                int in_idx = offset_from_index(in_view, i);
                out[out_idx] = in[in_idx];
        }
}