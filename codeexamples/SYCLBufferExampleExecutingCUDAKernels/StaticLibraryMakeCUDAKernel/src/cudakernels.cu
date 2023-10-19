#include <stdio.h>

/*
 * CUDA Kernel Device code
 * Simple message to the stdout.
 */
__global__ void kernelHelloWorld()
{
  printf("Hello from the Static Library CUDA kernel project\n");
}

/*
 * CUDA Kernel Device code
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void kernelVectorAdd( const int *vpAccA, const int *vpAccB, int *vpAccSum, int vnElements ) 
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if( i < vnElements ) 
  {
    vpAccSum[i] = vpAccA[i] + vpAccB[i];
  }
}