#ifndef SAFE_CALLS
#define SAFE_CALLS

#define NVRTC_SAFE_CALL(x)                                  \
do {                                                        \
  nvrtcResult result = x;                                   \
  if (result != NVRTC_SUCCESS) {                            \
    cout << "\nNVRTC call error: " #x " failed with error " \
         << nvrtcGetErrorString(result) << '\n';            \
    Rf_errorcall(R_NilValue, "Cuda runtime call produced error"); \
  }                                                         \
                                                            \
                                                            \
                                                            \
                                                            \
} while(0)

#define CUDA_SAFE_CALL(x)                                  \
do {                                                       \
  cudaThreadSynchronize();                                 \
  const char *msg;                                         \
  const char *str;                                         \
  CUresult result = x;                                     \
  if (result != CUDA_SUCCESS) {                            \
    cuGetErrorName(result, &msg);                          \
    cuGetErrorString(result, &str);                        \
    cout << "\nCUDA call error: " #x " failed with error " \
         << msg << '\n';                                   \
    Rf_errorcall(R_NilValue, "Cuda call produced error"); \
  }                                                        \
} while(0)

#define CU_SAFE_CALL(x)                                    \
do {                                                       \
  cudaThreadSynchronize();                                 \
  cudaError_t result = x;                                  \
  if (result != cudaSuccess) {                             \
    cout << "\nCUDA call error: " #x " failed with error " \
         << cudaGetErrorString(result) << '\n';            \
    Rf_errorcall(R_NilValue, "Cuda call produced error");  \
  }                                                        \
} while(0)

#endif
