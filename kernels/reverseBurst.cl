#ifndef CELLS_PER_COLUMN
	#error "CELLS_PER_COLUMN is not defined"
#endif

#ifndef NUM_COLUMNS
	#error "NUM_COLUMNS is not defined"
#endif

unsigned int random(int seed1, int seed2)
{
	uint seed = seed1 + get_global_id(0);
	uint t = seed ^ (seed << 11);
	return seed2 ^ (seed2 >> 19) ^ (t ^ (t >> 8));
}

//CELLS_PER_COLUMN: number of cells in each column
//NUM_COLUMNS: number of mini-coluumns
//seed1, seed2: seed for RNG
kernel void reverseBurst(global bool* x, int seed1, int seed2)
{
	int global_size = get_global_size(0);
	int global_id = get_global_id(0);
	uint rand_state = seed2;

	for(int i=global_id;i<NUM_COLUMNS;i+=global_size) {
		int sum = 0;
		for(int j=0;j<CELLS_PER_COLUMN;j++)
			sum += x[i*CELLS_PER_COLUMN+j];

		if(sum == CELLS_PER_COLUMN) {
			for(int j=0;j<CELLS_PER_COLUMN;j++)
				x[i*CELLS_PER_COLUMN+j] = 0;
			rand_state = random(seed1, seed2);
			x[i*CELLS_PER_COLUMN+(rand_state%CELLS_PER_COLUMN)] = 1;
		}
	}
}