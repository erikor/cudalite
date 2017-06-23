// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// jedit: :folding=explicit:
//
// cudalite.cpp: Lite weight interface between Cuda and R 
//
// Copyright (C) 2017 Eric Kort
//
// This file is part of cudalite.
//
// Cudalite is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// Rcpp is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Rcpp.  If not, see <http://www.gnu.org/licenses/>.

#include <Rcpp.h>
#include <string>
#include <streambuf>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <limits>
#include <typeinfo>
#include "safecalls.h"

using namespace Rcpp;
using namespace std;


/* cuData class
 *
 * A simple wrapper for CUdeviceptr objects to facilitate passing back 
 * and forth to R session and XPtr, as well as tracking data object 
 * dimensions. Handles data allocation on device as well as data transfer.
 * Note: there is currently no sanity checking to ensure there is room 
 * on device for data.
 */
class cuData {
 public:
  cuData(NumericMatrix);
  cuData(NumericVector);
  void freeMem();

  long length() {
    return len;
  }
  
  long nrow() {
    return rows;
  }

  long ncol() {
    return cols;
  }
 
  CUdeviceptr* getPtr() {
    return(&dp);
  }

  ~cuData() {
    if (dp) {
      cout << "Freeing device memory." << endl;
      CUDA_SAFE_CALL( cuMemFree(dp)); 
    }
  }

 private:
  long rows;
  long cols;
  long len;
  CUdeviceptr dp;
  void checkMem(size_t);
};

// free device memory
// device memory will be automatically freed via the destructor thanks to 
// the way XPtrs work with R's garbage collection mechanism.  However, explicit 
// memory freeing may be desired to make space for loading additional data.
void cuData::freeMem() {
  if (dp) {
    CUDA_SAFE_CALL( cuMemFree(dp)); 
  }
  dp = (CUdeviceptr)NULL;
}

// see how much available memory there is.
void cuData::checkMem(size_t size) {
  size_t free, total;
  CUDA_SAFE_CALL( cuMemGetInfo(&free, &total));
  if( free < size) {
    char * msg;
    int i = asprintf(&msg, "Insufficient memory available on device.  %zu bytes requested, %zu available\n", size, free);
    Rf_errorcall(R_NilValue, msg);
  }
}

// Load data from R Vector to device
cuData::cuData(NumericVector x){
  const size_t size = sizeof(double) * x.length();
  checkMem(size);
  len = x.length();
  rows = 0;
  cols = 0;
  
  CUDA_SAFE_CALL( cuMemAlloc(&dp, size)); 
  CUDA_SAFE_CALL( cuMemcpyHtoD(dp, REAL(x), size) );
}

// Load data from R Matrix to device
cuData::cuData(NumericMatrix x){
  const size_t size = sizeof(double) * size_t(x.nrow() * x.ncol());
  checkMem(size);
  len = 0;
  rows = x.nrow();
  cols = x.ncol();
  CUDA_SAFE_CALL( cuMemAlloc(&dp, size)); 
  CUDA_SAFE_CALL( cuMemcpyHtoD(dp, REAL(x), size) );
}

/* Cuda class
 * 
 * Light weight class to intialize device, keep track of environment, and provide 
 * methods for data transfer as well as loading and launching kernels.
 * 
 */
class Cuda {
  public:
    Cuda() {}
    double test();
    void loadKernel(string);
    void launchKernel(List, List, List);
    XPtr< cuData > h2dMatrix(NumericMatrix);
    XPtr< cuData > h2dVector(NumericVector);
    NumericVector d2hVector(SEXP);
    NumericMatrix d2hMatrix(SEXP);
    void dFree(SEXP);
    
  private: 
    CUfunction kernel;
    CUmodule module; 
    CUdevice cuDevice;
    CUcontext context;
    void checkCap(List, List);
};

/* See if device supports requested grid structure
 * 
 * @param grid    Three member list of x, y, and z dims of grid in blocks
 *                (List labels are optional)
 * @param block   Three member list of x, y, and z dims of blocks in threads
 *                (List labels are optional)
 * @return none   Called for side effect of gracefully throwing R error if 
 *                grid is too big for device.
 */
void Cuda::checkCap(List grid, List block) {
  dim3 dimG(*REAL(grid[0]), *REAL(grid[1]), *REAL(grid[2]));
  dim3 dimB(*REAL(block[0]), *REAL(block[1]), *REAL(block[2]));
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  bool ok = true;
  if(props.maxThreadsDim[0] < dimB.x || 
     props.maxThreadsDim[1] < dimB.y || 
     props.maxThreadsDim[2] < dimB.z)
    ok = false;
  if(props.maxThreadsPerBlock < dimB.x * dimB.y * dimB.z)
    ok = false;
  if(props.maxGridSize[0] < dimG.x ||
     props.maxGridSize[1] < dimG.y ||
     props.maxGridSize[2] < dimG.z)
    ok = false;
  
  if(!ok) {
    printf("\nMax Grid Size in blocks: %d(x), %d(y)\n", props.maxGridSize[0], props.maxGridSize[1]);
    printf("Max Block Size in threads: %d(x), %d(y)\n", props.maxThreadsDim[0], props.maxThreadsDim[1]);
    printf("Max total threads per block: %d\n", props.maxThreadsPerBlock);
    Rf_errorcall(R_NilValue, "GPU capabilities exceeded.  Reduce number of blocks or threads in grid.");
  }
  
}


/* Load and compile a kernel
* 
* @param fn    Filename (including path if not in current directory) containing source 
*              of kernel.  The kernel source code MUST define the function 'kernexec'  
*              because that is the entry point that will be called by launchKernel. 
* @return void Called for side effect of compiling and loading the kernel.  A pointer to 
*              the kernel is maintained as a private member of the object.
*/
void Cuda::loadKernel(string fn) {
  size_t ptxSize; 
  char *ptx;
  nvrtcProgram prog;
  
  ifstream file(fn.c_str());
  string buf((istreambuf_iterator<char>(file)),
             istreambuf_iterator<char>());
  NVRTC_SAFE_CALL( nvrtcCreateProgram(&prog, // prog 
                                      buf.c_str(), // buffer 
                                      "kernel", // 
                                      0, // numHeaders 
                                      NULL, // headers 
                                      NULL)); // includeNames
  
  NVRTC_SAFE_CALL( nvrtcCompileProgram(prog, 0, NULL));
  
  NVRTC_SAFE_CALL( nvrtcGetPTXSize(prog, &ptxSize)); 
  ptx = new char[ptxSize];
  NVRTC_SAFE_CALL( nvrtcGetPTX(prog, ptx));
  CUDA_SAFE_CALL( cuInit(0)); 
  CUDA_SAFE_CALL( cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL( cuCtxCreate(&context, 0, cuDevice);  
                    CUDA_SAFE_CALL( cuModuleLoadDataEx(&module, ptx, 0, 0, 0))); 
  CUDA_SAFE_CALL( cuModuleGetFunction(&kernel, module, "kernexec"));
  delete(ptx);
  return;
}

/* Free device memory
 * 
 * Device memory will be automatically freed via R's garbage collection 
 * mechanism.  However, if you need to free device memory immediately to make 
 * room for additional objects, you can free it explicitly with via this method.
 */
void Cuda::dFree(SEXP p) {
  XPtr< cuData > cudat(p);
  cudat->freeMem();
  return;
}  
  /* Launch a kernel
 * 
 * Assumes kernel has been loaded (see loadKernel) and necessary memory is 
 * allocated and, if necessary, populated on the device (see h2dVector, 
 * h2dMatrix)
 * 
 */
void Cuda::launchKernel(List grid, List block, List args) {
  dim3 dimG(*REAL(grid[0]), *REAL(grid[1]), *REAL(grid[2]));
  dim3 dimB(*REAL(block[0]), *REAL(block[1]), *REAL(block[2]));
  checkCap(grid, block);
  
  int n = args.length();
  void *a[args.length()];
  for(int i; i < n; i++) {
    SEXP p = args[i];
    switch(TYPEOF(p)) {
      case INTSXP: {
        a[i] = INTEGER(p);
        break;
      }
      case REALSXP: {
        // TODO: allow for conversion to single precision if desired
        a[i] = REAL(p);
        break;
      }
      case EXTPTRSXP: {
        XPtr< cuData > dp(p);
        a[i] = dp->getPtr();   
        break;
      }
      default: {
        Rf_errorcall(R_NilValue, "args must be list of numerics or pointers to device memory (see h2dvector and h2dmatrix).");
      }
    }
  }
  CUDA_SAFE_CALL( 
    cuLaunchKernel(kernel, 
                   dimG.x, dimG.y, dimG.z, // grid dim 
                   dimB.x, dimB.y, dimB.z,// block dim 
                   0, NULL, // shared mem and stream 
                   a, 0) // arguments
  ); 
  CUDA_SAFE_CALL(cuCtxSynchronize());
                                 
}


/* Load a numeric vector to device
 * 
 * @param x     The vector to load.
 * @return XPtr<cuData> An external point to the underlying cuData object.  This 
 *              can be used directly as part of the args argument to launchKernel.
 */
XPtr<cuData> Cuda::h2dVector(NumericVector x) {
  cuData *dat = new cuData(x);
  XPtr< cuData >  p(dat, true);
  CUDA_SAFE_CALL(cuCtxSynchronize());
  return p;
}

/* Load a numeric matrix to device
 * 
 * @param x     The matrix to load.
 * @return XPtr<cuData> An external point to the underlying cuData object.  This 
 *              can be used directly as part of the args argument to launchKernel.
 */
XPtr<cuData> Cuda::h2dMatrix(NumericMatrix x) {
  cuData *dat = new cuData(x);
  XPtr< cuData >  p(dat, true);
  CUDA_SAFE_CALL(cuCtxSynchronize());
  return p;
}

/* Retrieve data from the device
 * 
 * @param dx     The external pointer object containing the pointer to the data to retrieve.
 * @return NumericMatrix A matrix containing the data retrieved from the device.
 */
NumericMatrix Cuda::d2hMatrix(SEXP dx) {
  // TODO: verify argument type is EXTPTRSXP
  XPtr< cuData > cudat(dx);
  if(cudat->length() > 0) {
    Rf_errorcall(R_NilValue, "requested matrix from device but data appears to be a vector\n");
  }
  NumericMatrix hx(cudat->nrow(), cudat->ncol());
  size_t size = cudat->ncol() * cudat->nrow();
  CUDA_SAFE_CALL( cuMemcpyDtoH(hx.begin(), *(cudat->getPtr()), size*sizeof(double)) );
  return(hx);
}

/* Retrieve data from the device
 * 
 * @param dx     The external pointer object containing the pointer to the data to retrieve.
 * @return NumericVector A vector containing the data retrieved from the device.
 */
NumericVector Cuda::d2hVector(SEXP dx) {
  // TODO: verify argument type is EXTPTRSXP
  XPtr< cuData > cudat(dx);
  if(cudat->nrow() >  0) {
    Rf_errorcall(R_NilValue, "requested vector from device but data appears to be a matrix\n");
  }
  NumericVector hx(cudat->length());
  size_t size = cudat->length();
  CUDA_SAFE_CALL( cuMemcpyDtoH(hx.begin(), *(cudat->getPtr()), size*sizeof(double)) );
  return(hx);
}


RCPP_MODULE(cuda) {
  class_<Cuda>( "Cuda" )
  .constructor()
  .method( "loadKernel", &Cuda::loadKernel, "Read kernel from source file, compile, and load it on device." )
  .method( "launchKernel", &Cuda::launchKernel, "Launch a loaded kernel with provided arguments." )
  .method( "h2dMatrix", &Cuda::h2dMatrix, "Load numeric matrix to device." )
  .method( "h2dVector", &Cuda::h2dVector, "Load numeric vector to device." )
  .method( "d2hMatrix", &Cuda::d2hMatrix, "Retrieve numeric matrix from device." )
  .method( "d2hVector", &Cuda::d2hVector, "Retrieve numeric vector from device." )
  ;
}