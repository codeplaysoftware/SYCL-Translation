// Local headers:
#include "staticlibfnscudakernel.h"
#include "cudakernels.cu"

// Third party headers:
#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

namespace StaticLibFnsCUDAKernel
{
    void HelloWorldFacade()
    {
        kernelHelloWorld<<< 1, 1 >>>();
        
        // Interop with host_task doesn't add CUDA event to task graph
        // so we must manually sync here.
        cudaDeviceSynchronize();
    }

    cudaError_t KernelCUDAVectorAddFacade( const int vBlocksPerGrid, const int vThreadsPerBlock, const int *vpPtrAccA, const int *vpPtrAccB, int *vpPtrAccSum, const int vNElements )
    {
        kernelVectorAdd<<< vBlocksPerGrid, vThreadsPerBlock >>>(vpPtrAccA, vpPtrAccB, vpPtrAccSum, vNElements );
       
        // Interop with host_task doesn't add CUDA event to task graph
        // so we must manually sync here.
        cudaDeviceSynchronize();

        return cudaGetLastError();
    }
} // namespace StaticLibFnsCUDAKernel
