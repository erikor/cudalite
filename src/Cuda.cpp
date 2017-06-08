#include <Rcpp.h>
#include <string>
#include <streambuf>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

using namespace Rcpp;
using namespace std;

#define NVRTC_SAFE_CALL(x)                                  \
do {                                                        \
  nvrtcResult result = x;                                   \
  if (result != NVRTC_SUCCESS) {                            \
    cerr << "\nNVRTC call error: " #x " failed with error " \
         << nvrtcGetErrorString(result) << '\n';            \
    exit(1);                                                \
  }                                                         \
} while(0)

#define CUDA_SAFE_CALL(x)                                  \
do {                                                       \
  const char *msg;                                         \
  CUresult result = x;                                     \
  if (result != CUDA_SUCCESS) {                            \
    cuGetErrorName(result, &msg);                          \
    cerr << "\nCUDA call error: " #x " failed with error " \
         << msg << '\n';                                   \
    exit(1);                                               \
  }                                                        \
} while(0)

#define CU_SAFE_CALL(x)                                                           \
do {                                                                              \
  const char *msg;                                                                \
  cudaError_t result = x;                                                         \
  if (result != cudaSuccess) {                                                    \
    cerr << "\nCUDA call error: " #x " failed with error "                        \
         << cudaGetErrorString(result) << '\n';                                   \
    exit(1);                                                                      \
  }                                                                               \
} while(0)


class Cuda {
  public:
    Cuda() {}
    double test();
    void loadKernel(string fn);
    void loadMatrix(NumericMatrix x);

  private: 
    CUfunction _kernel;
};

void Cuda::loadKernel(string fn) {
  size_t ptxSize; 
  char *ptx;
  nvrtcProgram prog;
  CUmodule module; 
  CUfunction kernel; 
  CUdevice cuDevice;
  CUcontext context;
  
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
  CUDA_SAFE_CALL( cuCtxCreate(&context, 0, cuDevice);  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0))); 
  CUDA_SAFE_CALL( cuModuleGetFunction(&kernel, module, "kernexec"));
  CUDA_SAFE_CALL( cuModuleGetFunction(&kernel, module, "kernexec"));
  delete(ptx);
  
  _kernel = kernel;
}

void Cuda::loadMatrix(NumericMatrix x) {
  int* device;
  const size_t size = sizeof(float) * size_t(x.nrow() * x.ncol());
  CU_SAFE_CALL( cudaMalloc((void **)&device, size)); 
  CU_SAFE_CALL( cudaMemcpy(device, REAL(x), size, cudaMemcpyHostToDevice)); 
  //kernel<<<N,N>>>(a_device); 
  return;
}


RCPP_MODULE(unif) {
  class_<Cuda>( "Cuda" )
//  .constructor<double,double>()
  .constructor()
  .method( "loadKernel", &Cuda::loadKernel )
  .method( "loadMatrix", &Cuda::loadMatrix )
;
}