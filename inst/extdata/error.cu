extern "C" 

__global__
void kernexec(int n, float a, float *x, float *y)
{
    xxxx
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          y[i] = a * x[i] + y[i];
      }
}
