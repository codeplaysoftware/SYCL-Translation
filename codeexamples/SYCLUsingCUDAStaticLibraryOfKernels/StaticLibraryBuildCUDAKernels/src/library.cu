// In-house headers:
#include "staticlibfnscudakernelsDXTC.h"
#include "cudakernels.cu"

namespace StaticLibFnsCUDAKernelsDXTC
{
    cudaError_t CompressFacade( const int vBlocksPerLaunch, const uint vBlocks, 
                                const uint *vpPermutations, const uint *vpImage, uint *vpResult, const int vBlockOffset ) 
    {
        compress<<< min( vBlocksPerLaunch, vBlocks - vBlockOffset ), NUM_THREADS >>>(
              vpPermutations, vpImage, (uint2 *) vpResult, vBlockOffset );

        // Interop with host_task doesn't add CUDA event to task graph
        // so we must manually sync here.
        cudaDeviceSynchronize();

        return cudaGetLastError();
    }

} // namespace StaticLibFnsCUDAKernelsDXTC
