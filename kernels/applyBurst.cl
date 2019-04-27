#ifndef CELLS_PER_COLUMN
	#error "CELLS_PER_COLUMN is not defined"
#endif

#ifndef NUM_COLUMNS
	#error "NUM_COLUMNS is not defined"
#endif

kernel void applyBurst(global bool* restrict x, global bool* restrict y)
{
	int global_size = get_global_size(0);
	int global_id = get_global_id(0);
	for(int i=global_id;i<NUM_COLUMNS;i+=global_size) {
		int fill_value = -1;
		if(x[i] == 0)
			fill_value = 0;
		else {
			int sum = 0;
			for(int j=0;j<CELLS_PER_COLUMN;j++)
				sum += y[i*CELLS_PER_COLUMN+j];
			if(sum == 0)
				fill_value = 1;
		}

		#pragma unroll 8
		for(int j=0;j<CELLS_PER_COLUMN && fill_value != -1;j++)
			y[i*CELLS_PER_COLUMN+j] = fill_value;
	}
}