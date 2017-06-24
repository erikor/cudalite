## Synopsis

A simple interface between R and CUDA providing facilties for data transfer and kernel launching.

## Motivation

There is already a package for interfacing R and CUDA, the aptly named RCUDA which provides an elegant and comprehensive interface to CUDA computation.  That package likely would meet your needs.  However, I decided to develop the `cudalite` package for a couple of reasons.  First, RCUDA did not compile out of the box for me.  (Although I have since been able to resolve those issues, and I desribe the fixes in issues raised at the RCUDA github repository.)  Second, I wanted a simple, relatively low level (and therefore highly flexible) interface to CUDA. Finally, I wanted to have a solid understanding of the fundamentals of CUDA and how data was flowing back and forth between R, the host, and the device.

Thus, `cudalite` was born.  I hope you find it useful and simple to use.  If you would rather use RCUDA, by all means do so.

## Installation

To build, install, and use the `cudalite` package you will need to have the [CUDA SDK](https://developer.nvidia.com/cuda-downloads) installed and a CUDA capable GPU available.  

Assuming those requirements are met, the package can be installed with in an R session as follows:

```bash
devtools::install_git("https://github.com/erikor/cudalite")
```

The package includes a configure script that will do its best to discern the location of your CUDA installation.

## Code Example

We will work with the following simple kernel that just copies one matrix to another on the device.  In this example, it is assumed this kernel is saved as "mykernel.cu"

```c
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
    for (int i = r; i < nrow; i+= blockDim.y * gridDim.y) {
      for (int j = c; j < ncol; j+= blockDim.x * gridDim.x) {
        // RCpp's numeric matrix stores data column-wise
        int index = i + nrow * j;
        out[index] = x[index];
      }
    }
}

```

Note that the entry point to this kernel is "kernexec".  This is **required**.  When the kernel is launched, `cudalite` will call the `kernexec` function in the kernel.  In the future, we may make naming more flexible.  

Now we can compile the kernel from with an R session and load it to the device:

```
library(`cudalite`)
cu <- new(Cuda)
cu$loadKernel("mykernel.cu")
```

Now we need to load the data the kernel requires. 

```{r eval=FALSE}
x <- matrix(rnorm(20), nrow=5, ncol=4)
y <- matrix(0, nrow=5, ncol=4)
xp <- cu$h2dMatrix(x);
yp <- cu$h2dMatrix(y);
```

The function `h2dMatrix` loads the data on device and returns a pointer to it.  More precisely, it return a pointer to an object that encapsulates the device pointer along with some descriptive information about the original data structure so it can be retrieved later if desired.

Now we are ready to launch the kernel.
```{r eval=FALSE}
cu$launchKernel(list(1,1,1), list(3,2,1), list(5, 4, xp, yp))
```

The first argument to `launchKernel` is list of three elements that gives the x, y, and z dimensions of the GPU grid (in *blocks*).  The second list gives th x, y, and z dimensions of each block within the grid (in *threads*).  The number of blocks and size of each block depend on a few factors. First of all, the number of threads available on the GPU, though much larger than your CPU, is still finite.  Second, there are limitations on the number of threads per block, and the number of blocks per grid.  This varies between different models of GPUs.

Fortunately, if you exceed these limits, `cudalite` will tell you and you can try again with a smaller grid or smaller blocks.

In general, the most efficient computation is likely achieved by having a grid that is the same size as your matrix, or divides evenly into your matrix.  If your grid is much larger than your matrix, you are wasting threads.  If your grid is much smaller than your matrix, your computation will take longer since the kernel will need to stride accross the data matrix to process all of it.  On the other hand, if your matrix is larger than the largest grid your GPU can provide, you will of course need to stride.  (Note that the kernel above strides automatically based on the size of the data matrix and the dimensions of the grid).

The final argument given to launchKernel is a list of arguments that the kernel expects, in the order the kernel expects them.  Simple numeric values can be provided directly, where as pointers to device memory are given by the values returned by `h2dMatrix`.

Finally, we can retrieve our data and check that the kernel copied our data as expected.

```{r eval=FALSE}
res <- cu$d2hMatrix(yp)
all.equal(x, res)
```
