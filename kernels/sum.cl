#ifndef InType
        #error InType not defined
#endif

#ifndef OutType
        #error OutType not defined
#endif

kernel void sum(global InType* restrict x, global OutType* restrict y, int in_size, int chunk_size)
{
        int global_size = get_global_size(0);
        int global_id = get_global_id(0);

        int problem_size = in_size/chunk_size;
        for(int i=global_id;i<problem_size;i+=global_size) {
                OutType s = 0;
                for(int j=0;j<chunk_size;j++)
                        s += x[i*chunk_size+j];
                y[i] = s;
        }
}