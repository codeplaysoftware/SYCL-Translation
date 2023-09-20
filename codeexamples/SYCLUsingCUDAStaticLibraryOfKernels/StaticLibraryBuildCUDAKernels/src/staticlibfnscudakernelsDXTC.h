namespace StaticLibFnsCUDAKernelsDXTC
{
    // List of CUDA kernels (via facade wrappers) functions
    cudaError_t CompressFacade( const int vBlocksPerLaunch, const uint vBlocks, 
                                const uint *vpPermutations, const uint *vpImage, uint *vpResult, const int vBlockOffset );

} // namespace StaticLibFnsCUDAKernelsDXTC