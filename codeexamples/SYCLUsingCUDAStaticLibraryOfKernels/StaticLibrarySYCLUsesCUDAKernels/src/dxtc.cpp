/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Utilities and system includes:
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// CUDA toolkit utilities
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>

// In-house headers:
#include "dds.h"
#include "permutations.h"
#include "staticlibfnscudakernelsDXTC.h"

// Definitions:
#define INPUT_IMAGE "./../data/teapot512_std.ppm"
#define REFERENCE_IMAGE "./../data/teapot512_ref.dds"
#define ERROR_THRESHOLD 0.02f
#define NUM_THREADS 64  // Number of threads per block.
#define NUM_PERMUTATIONS 1024

// Helper structs and functions to validate the output of the compressor.
// We cannot simply do a bitwise compare, because different compilers produce
// different results for different targets due to floating point arithmetic.
union Color32 {
  struct {
    unsigned char b, g, r, a;
  };
  unsigned int u;
};

union Color16 {
  struct {
    unsigned short b : 5;
    unsigned short g : 6;
    unsigned short r : 5;
  };
  unsigned short u;
};

struct BlockDXT1 {
  Color16 col0;
  Color16 col1;
  union {
    unsigned char row[4];
    unsigned int indices;
  };

  void decompress(Color32 colors[16]) const;
};

void BlockDXT1::decompress(Color32 *colors) const {
  Color32 palette[4];

  // Does bit expansion before interpolation.
  palette[0].b = (col0.b << 3) | (col0.b >> 2);
  palette[0].g = (col0.g << 2) | (col0.g >> 4);
  palette[0].r = (col0.r << 3) | (col0.r >> 2);
  palette[0].a = 0xFF;

  palette[1].r = (col1.r << 3) | (col1.r >> 2);
  palette[1].g = (col1.g << 2) | (col1.g >> 4);
  palette[1].b = (col1.b << 3) | (col1.b >> 2);
  palette[1].a = 0xFF;

  if (col0.u > col1.u) {
    // Four-color block: derive the other two colors.
    palette[2].r = (2 * palette[0].r + palette[1].r) / 3;
    palette[2].g = (2 * palette[0].g + palette[1].g) / 3;
    palette[2].b = (2 * palette[0].b + palette[1].b) / 3;
    palette[2].a = 0xFF;

    palette[3].r = (2 * palette[1].r + palette[0].r) / 3;
    palette[3].g = (2 * palette[1].g + palette[0].g) / 3;
    palette[3].b = (2 * palette[1].b + palette[0].b) / 3;
    palette[3].a = 0xFF;
  } else {
    // Three-color block: derive the other color.
    palette[2].r = (palette[0].r + palette[1].r) / 2;
    palette[2].g = (palette[0].g + palette[1].g) / 2;
    palette[2].b = (palette[0].b + palette[1].b) / 2;
    palette[2].a = 0xFF;

    palette[3].r = 0x00;
    palette[3].g = 0x00;
    palette[3].b = 0x00;
    palette[3].a = 0x00;
  }

  for (int i = 0; i < 16; i++) {
    colors[i] = palette[(indices >> (2 * i)) & 0x3];
  }
}

static int compareColors(const Color32 *b0, const Color32 *b1) {
  int sum = 0;
  for (int i = 0; i < 16; i++) {
    int r = (b0[i].r - b1[i].r);
    int g = (b0[i].g - b1[i].g);
    int b = (b0[i].b - b1[i].b);
    sum += r * r + g * g + b * b;
  }
  return sum;
}

static int compareBlock(const BlockDXT1 *b0, const BlockDXT1 *b1) {
  Color32 colors0[16];
  Color32 colors1[16];

  if (memcmp(b0, b1, sizeof(BlockDXT1)) == 0) {
    return 0;
  } else {
    b0->decompress(colors0);
    b1->decompress(colors1);
    return compareColors(colors0, colors1);
  }
}

cudaError_t CompressUsingCUDA( sycl::queue &vQ, const int vBlocksPerLaunch, const uint vBlocks,
                        const uint *vpd_permutations, const uint *vpd_data, uint *vpd_result, 
                        const int vBlockOffset )
{
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Submit a command group to the queue by a C++ function that contains the
  // data access permission and device computation (kernel).
  vQ.submit( [&]( sycl::handler &vH )
  { 
    vH.host_task( [=]( const sycl::interop_handle &vIh ) 
    // Sets the maximum sub-group size, not the size of all sub-groups in the dispatch
    [[sycl::reqd_sub_group_size(32)]] 
    {
      cudaError_t *pErr = const_cast< cudaError_t * >( &err );
      *pErr = StaticLibFnsCUDAKernelsDXTC::CompressFacade( vBlocksPerLaunch, vBlocks,
                                      vpd_permutations, vpd_data, vpd_result, vBlockOffset );
    });
  });

  return err;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
try 
{
  cudaError_t errorCudaStatus = cudaSuccess;
  const auto deviceSelector = sycl::gpu_selector_v;
  sycl::queue queue( deviceSelector );
  
  std::cout << argv[0] << " Starting...\n";
  std::cout << "Running on device: " << queue.get_device().get_info< sycl::info::device::name >() << "\n\n";

  // Load input image.
  unsigned char *image_data = NULL;
  uint W, H;
  char *image_path = sdkFindFilePath(INPUT_IMAGE, argv[0]);
  if (image_path == 0) {
    printf("Error, unable to find source image  <%s>\n", image_path);
    exit(EXIT_FAILURE);
  }
  printf("Using image: %s \n", image_path);
  if (!sdkLoadPPM4ub(image_path, &image_data, &W, &H)) {
    printf("Error, unable to open source image file <%s>\n", image_path);
    exit(EXIT_FAILURE);
  }
  uint w = W, h = H;
  printf("Image Loaded '%s', w:%d x h:%d pixels\n\n", image_path, w, h);

  // Allocate input image.
  const uint memSize = w * h * 4;
  assert(0 != memSize);
  uint *block_image = (uint *)malloc(memSize);
  if( block_image == 0 )
  {
     printf( "Error, unable to allocate host memory for image w * h * uint of %d size\n", memSize );
     exit( EXIT_FAILURE );
  }

  // Convert linear image to block linear.
  for (uint by = 0; by < h / 4; by++) {
    for (uint bx = 0; bx < w / 4; bx++) {
      for (int i = 0; i < 16; i++) {
        const int x = i & 3;
        const int y = i / 4;
        block_image[(by * w / 4 + bx) * 16 + i] =
            ((uint *)image_data)[(by * 4 + y) * 4 * (W / 4) + bx * 4 + x];
      }
    }
  }

  // The image data.
  /* DPCT_ORIG checkCudaErrors(cudaMalloc((void **)&d_data, memSize));*/
  uint *d_data = (uint *) sycl::malloc_device( memSize, queue );
  if( d_data == 0 )
  {
     printf( "Error, unable to allocate device memory for image of %d size\n", memSize );
     exit( EXIT_FAILURE );
  }

  // Image results.
  const uint compressedSize = (w / 4) * (h / 4) * 8;
  /* DPCT_ORIG checkCudaErrors(cudaMalloc((void **)&d_result, compressedSize));*/
  uint *d_result = (uint *) sycl::malloc_device( compressedSize, queue );
  if( d_result == 0 )
  {
     printf( "Error, unable to allocate device memory for result of %d size\n", compressedSize );
     exit( EXIT_FAILURE );
  }
  uint *h_result = (uint *)malloc(compressedSize);
  if( h_result == 0 )
  {
     printf( "Error, unable to allocate host memory for result of %d size\n", compressedSize );
     exit( EXIT_FAILURE );
  }

  /* DPCT_ORIG checkCudaErrors(cudaMalloc((void **)&d_permutations, NUM_PERMUTATIONS * sizeof(uint)));*/
  uint *d_permutations = sycl::malloc_device< uint >( NUM_PERMUTATIONS, queue );
  if( d_permutations == 0 )
  {
     printf( "Error, unable to allocate device memory for permutations of %d size\n", NUM_PERMUTATIONS );
     exit( EXIT_FAILURE );
  }
  
  // Compute permutations.
  uint permutations[ NUM_PERMUTATIONS ];
  computePermutations(permutations);
  // Copy permutations host to device.
  /* DPCT_ORIG checkCudaErrors(cudaMemcpy(d_permutations, permutations, NUM_PERMUTATIONS * sizeof(uint), cudaMemcpyHostToDevice));*/
  queue.memcpy( d_permutations, permutations, NUM_PERMUTATIONS * sizeof( uint ) ).wait();
  
  // Copy image from host to device.
  /* DPCT_ORIG cudaMemcpy(d_data, block_image, memSize, cudaMemcpyHostToDevice));*/
  queue.memcpy( d_data, block_image, memSize ).wait();

  // Create a timer.
  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  // Determine launch configuration and run timed computation numIterations
  // times. Rounds up by 1 block in each dim if %4 != 0
  uint blocks = ((w + 3) / 4) * ((h + 3) / 4);  

  // Get number of SMs on this GPU
  /* DPCT_ORIG cudaDeviceProp deviceProp;*/
  dpct::device_info deviceProp;
  /* DPCT_ORIG checkCudaErrors(cudaGetDevice(&devID));*/
  int devID = dpct::dev_mgr::instance().current_device_id();
  dpct::dev_mgr::instance().get_device(devID).get_device_info(deviceProp);

  // Restrict the numbers of blocks to launch on low end GPUs to avoid kernel timeout
  /* DPCT_ORIG int blocksPerLaunch = min(blocks, 768 * deviceProp.multiProcessorCount);*/
  int blocksPerLaunch = std::min<uint>(blocks, 768 * deviceProp.get_max_compute_units());

  printf("Running DXT Compression on %u x %u image...\n", w, h);
  printf("\n%u Blocks, %u Threads per Block, %u Threads in Grid...\n\n", blocks, NUM_THREADS, blocks * NUM_THREADS);
  
  int numIterations = 1;
  for (int i = -1; i < numIterations; ++i) {
    if (i == 0) 
    {
      /* DPCT_ORIG checkCudaErrors(cudaDeviceSynchronize());*/
      queue.wait_and_throw();
      sdkStartTimer(&timer);
    }

    for (int j = 0; j < (int)blocks; j += blocksPerLaunch) {
      /* DPCT_ORIG compress<<<min(blocksPerLaunch, blocks - j), NUM_THREADS>>>(d_permutations, d_data, (uint2 *)d_result, j);*/
      errorCudaStatus = CompressUsingCUDA( queue, blocksPerLaunch, blocks, d_permutations, d_data, d_result, j );
      if( errorCudaStatus != cudaSuccess )  
      { 
        fprintf( stderr, "Failed to launch KernelCUDAVectorAddFacade (error code %s)!\n", cudaGetErrorString( errorCudaStatus ) );
      }   
    }
  }

  // Sync to host, stop timer, record perf
  /* DPCT_ORIG checkCudaErrors(cudaDeviceSynchronize());*/
  queue.wait_and_throw();

  sdkStopTimer(&timer);
  double dAvgTime = 1.0e-3 * sdkGetTimerValue(&timer) / (double)numIterations;
  printf("dxtc, Throughput = %.4f MPixels/s, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %d\n",
      (1.0e-6 * (double)(W * H) / dAvgTime), dAvgTime, (W * H), 1, NUM_THREADS);

  // Copy result data from device to host
  queue.memcpy( h_result, d_result, compressedSize ).wait();

  // Write out result data to DDS file
  char output_filename[1024];
  strcpy(output_filename, image_path);
  strcpy(output_filename + strlen(image_path) - 3, "dds");
  FILE *fp = fopen(output_filename, "wb");
  if (fp == 0) {
    printf("Error, unable to open output image <%s>\n", output_filename);
    exit(EXIT_FAILURE);
  }
  DDSHeader header;
  header.fourcc = FOURCC_DDS;
  header.size = 124;
  header.flags = (DDSD_WIDTH | DDSD_HEIGHT | DDSD_CAPS | DDSD_PIXELFORMAT | DDSD_LINEARSIZE);
  header.height = h;
  header.width = w;
  header.pitch = compressedSize;
  header.depth = 0;
  header.mipmapcount = 0;
  memset(header.reserved, 0, sizeof(header.reserved));
  header.pf.size = 32;
  header.pf.flags = DDPF_FOURCC;
  header.pf.fourcc = FOURCC_DXT1;
  header.pf.bitcount = 0;
  header.pf.rmask = 0;
  header.pf.gmask = 0;
  header.pf.bmask = 0;
  header.pf.amask = 0;
  header.caps.caps1 = DDSCAPS_TEXTURE;
  header.caps.caps2 = 0;
  header.caps.caps3 = 0;
  header.caps.caps4 = 0;
  header.notused = 0;
  fwrite(&header, sizeof(DDSHeader), 1, fp);
  fwrite(h_result, compressedSize, 1, fp);
  fclose(fp);

  // Make sure the generated image is correct.
  const char *reference_image_path = sdkFindFilePath(REFERENCE_IMAGE, argv[0]);
  if (reference_image_path == 0) {
    printf("Error, unable to find reference image\n");
    exit(EXIT_FAILURE);
  }
  fp = fopen(reference_image_path, "rb");
  if (fp == 0) {
    printf("Error, unable to open reference image\n");
    exit(EXIT_FAILURE);
  }
  fseek(fp, sizeof(DDSHeader), SEEK_SET);
  uint referenceSize = (W / 4) * (H / 4) * 8;
  uint *reference = (uint *)malloc(referenceSize);
  fread(reference, referenceSize, 1, fp);
  fclose(fp);

  printf("\nChecking accuracy...\n");
  float rms = 0;
  for (uint y = 0; y < h; y += 4) {
    for (uint x = 0; x < w; x += 4) {
      uint referenceBlockIdx = ((y / 4) * (W / 4) + (x / 4));
      uint resultBlockIdx = ((y / 4) * (w / 4) + (x / 4));
      int cmp = compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx,
                             ((BlockDXT1 *)reference) + referenceBlockIdx);
      if (cmp != 0.0f) {
        printf("Deviation at (%4d,%4d):\t%f rms\n", x / 4, y / 4, float(cmp) / 16 / 3);
      }
      rms += cmp;
    }
  }
  rms /= w * h * 3;
  printf("RMS(reference, result) = %f\n\n", rms);
  printf(rms <= ERROR_THRESHOLD ? "Test passed\n" : "Test failed!\n");
    
  // Free allocated resources and exit
/* DPCT_ORIG   checkCudaErrors(cudaFree(d_permutations));*/
  if( d_permutations != 0 ) sycl::free( d_permutations, queue );
/* DPCT_ORIG   checkCudaErrors(cudaFree(d_data));*/
  if( d_data != 0 ) sycl::free( d_data, queue );
/* DPCT_ORIG   checkCudaErrors(cudaFree(d_result));*/
  if( d_result != 0 ) sycl::free( d_result, queue );
  if( image_path != 0 ) free( image_path );
  if( image_data != 0 ) free( image_data );
  if( d_result != 0 ) free( block_image );
  if( h_result != 0 ) free( h_result );
  if( reference != 0 ) free( reference );
  if( timer != 0 ) sdkDeleteTimer( &timer );

  return EXIT_SUCCESS; 
}
catch (sycl::exception const &exc) 
{
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
