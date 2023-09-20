namespace StaticLibFnsCUDAKernel
{
    void HelloWorldFacade();
    cudaError_t KernelCUDAVectorAddFacade( const int vBlocksPerGrid, const int vThreadsPerBlock, const int *vpPtrAccA, const int *vpPtrAccB, int *vpPtrAccSum, const int vNElements );
    
} // namespace StaticLibFnsCUDAKernel