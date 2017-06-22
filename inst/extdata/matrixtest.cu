extern "C" 

__global__
void kernexec(double n, double a, double *x, double *y, double *out)
{
    // striding to allow for matrix larger than number of available threads
    // https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    //
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          out[i] = a * x[i] + y[i];
          printf("index %d: x: %f, y: %f, value: %f\n", i, x[i], y[i], out[i]);
      }
}
