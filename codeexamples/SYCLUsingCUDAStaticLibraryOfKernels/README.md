# VSCodeMakeStaticLibraryCUDAkernelsExampleDXTC

This project shows how to call a CUDA __global__ kernel from within a SYCL program using one task graph. The global kernel makes further to kernel calls to CUDA __device__ functions. The program is version of the original CUDA sample **dxtc** found in the NVIDIA CUDA samples github repository [here](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/dxtc).

DXTC stands for High Quality DirectX Compressor. This example uses the same framework (a SYCL framework) and CUDA kernel functions as the original.

This SYCL example is based on the SYCL USM framework model. Refer to the Codeplay CUDA interopability with SYCL example **VSCodeMakeStaticLibraryCUDAkernels** to see how a SYCL buffer framework model is used. Unlike the **VSCodeMakeStaticLibraryCUDAkernels** example this example does not mix SYCL kernel execution with CUDA kernels in the same task graph (the one queue) to act on one set of data to create some new data or results.

## CUDA interopability approach with SYCL
All the CUDA kernels are put in a static library. The global kernel is wrapped by a C++ Facade function. This library is compiled with the NVIDIA compiler.

The static library is linked to a SYCL project. The Facade wrapper function is called from within a SYCL host_task() fucntion, which in turn is enqueued in a SYCL queue.

## Build instructions
Ensure the CUDA samples **Common** directory is included in the makefile for the static library build. The INCLUDE_PATH is neccesary as the CUDA code pulls in various helper files, i.e., helper_functions.h, in order to compile successfully.

The same CUDA samples **Common** directory needs also to be included in the VSCode project **StaticLibrarySYCLUsesCUDAKernels** to build the SYCL code with the CUDA static library. Edit the line "-I${workspaceFolder}/< change path to >/cuda-samples-nvidia/Common" in each of the build configurations in the project's task.json file.

Next, build the static library which contains the CUDA kernels. 
Using Microsoft's Visual Studio Code IDE (VSCode), open the VSCode project folder *VSCodeStaticLibraryMakeCUDAKernesDXTC*. Select the **task.json** file in the **.vscode** directory and use key shift+alt+b to use the default build task. The build task uses the makefile to build the static library in the **bin** directory.

Once the static library is built, it is linked to the main program to enable the SYCL program to call upon the CUDA kernels. Open up a new project window using VSCode for the folder *VSCodeStaticLibraryUse*. Use shift+alt+b to select an **icpx** compiler build and hit return. The main project will build putting the executable in the bin directory.

## Development environment
Ubuntu 22.04 LTS \
Intel oneAPI Base Toolkit 2023.1 + CUDA Plugin 2023.1 (setenvars.sh is active) \
NVIDIA CUDA Toolkit 12.0 \
NVIDIA CUDA Samples Common directory for CUDA helper include files (/cuda-samples-nvidia/Common) \
Compilers: NVIDIA's nvcc and Intel's icpx \
Target device: NVIDIA GTX2060 \
Development IDE: Microsoft VSCode + extensions: Microsoft C/C++ pack, NVIDIA Nsight
