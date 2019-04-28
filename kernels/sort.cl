
#ifndef MAX_SYNAPSE_PER_CELL
	#error "MAX_SYNAPSE_PER_CELL not defined"
#endif

void merge(global unsigned int* restrict a1, global float* restrict a2, int l, int m, int r
	, global unsigned int* restrict aux_buffer1, global float* restrict aux_buffer2)
{
	int n1 = m - l + 1;
	int n2 =  r - m;

	global unsigned int* restrict L = aux_buffer1;
	global unsigned int* restrict R = aux_buffer1+n1;
	global float* restrict L2 = aux_buffer2;
	global float* restrict R2 = aux_buffer2+n1;

	for (int i=0;i<n1;i++) {
		L[i] = a1[l+i];
		L2[i] = a2[l+i];
	}
	for (int j=0; j<n2;j++) {
		R[j] = a1[m+1+j];
		R2[j] = a2[m+1+j];
	}

	int i, j, k;
	i = 0;
	j = 0;
	k = l;
	while (i < n1 && j < n2) {
		if (L[i] <= R[j]) {
			a1[k] = L[i];
			a2[k] = L2[i];
			i++;
		}
		else {
			a1[k] = R[j];
			a2[k] = R2[j];
			j++;
		}
		k++;
	}

	while (i < n1) {
		a1[k] = L[i];
		a2[k] = L2[i];
		i++;
		k++;
	}

	while (j < n2) {
		a1[k] = R[j];
		a2[k] = R2[j];
		j++;
		k++;
	}
}

void mergeSort(global unsigned int* restrict connections, global float* restrict permeances, int n
	,global unsigned int* restrict aux_buffer1, global float* restrict aux_buffer2)
{
	for (int curr_size=1; curr_size<=n-1; curr_size = 2*curr_size) {
	   for (int left_start=0; left_start<n-1; left_start += 2*curr_size) {
		   int mid = left_start + curr_size - 1;
		   int right_end = min(left_start + 2*curr_size - 1, n-1);
		   merge(connections, permeances, left_start, mid, right_end, aux_buffer1, aux_buffer2);
		}
	}
}

//CELLS_PER_COLUMN: number of cells in each column
//aux_buffer: Buffer for tempory storage when sorting, must be the sizeof(int)*global_size[0]
kernel void sortSynapse(global unsigned int* restrict connections, global float* restrict permeances, int num_cells
	, global unsigned int* restrict aux_buffer1, global float* restrict aux_buffer2)
{
	int global_id = get_global_id(0);
	int global_size = get_global_size(0);

	for(int i=global_id;i<num_cells;i+=global_size) {
		int offset = i*MAX_SYNAPSE_PER_CELL;
		mergeSort(connections+offset, permeances+offset, MAX_SYNAPSE_PER_CELL, aux_buffer1+offset, aux_buffer2+offset);
	}
}