extern "C" 

#include <stdio.h>

__global__
void kernexec(double n)
{
    printf("Received argument %f at block %d, thread %d %d %d\n", n, blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z);
}
