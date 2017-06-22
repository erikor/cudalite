extern "C" 

__global__
void kernexec(double nrow, double ncol, double *x, double *out)
{

   /*
    * striding to allow for matrix larger than number of available threads
    *
    * should produce exact copy of input matrix regardless of size of available device grid
    * inspired by 1D example at: 
    * https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    */

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    for (int i = r; i < nrow; i+= blockDim.x * gridDim.x) {
      for (int j = c; j < ncol; j+= blockDim.y * gridDim.y) {
        int index = i * ncol + j;
        out[index] = x[index];
      }
    }
}
