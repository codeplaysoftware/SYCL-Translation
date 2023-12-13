# A SYCL program executing CUDA kernels

This project shows how to call a CUDA __global__ kernel from within a SYCL program using one task graph. The global kernel makes further to kernel calls to CUDA __device__ functions. The program is version of the original CUDA sample **dxtc** found in the NVIDIA CUDA samples github repository [here](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/dxtc).

DXTC stands for High Quality DirectX Compressor. This example uses the same framework (a SYCL framework) and CUDA kernel functions as the original.

This SYCL example is based on the SYCL USM framework model. Refer to the Codeplay CUDA interopability with SYCL example **SYCLBufferUsingStaticLibraryCUDAkernels** to see how a SYCL buffer framework model is used. Unlike the **SYCLBufferUsingStaticLibraryCUDAkernels** example this example does not mix SYCL kernel execution with CUDA kernels in the same task graph (the one queue) to act on one set of data to create some new data or results.

## CUDA interopability approach with SYCL
All the CUDA kernels are put in a static library. The global kernel is wrapped by a C++ Facade function. This library is compiled with the NVIDIA compiler.

The static library is linked to a SYCL project. The Facade wrapper function is called from within a SYCL host_task() fucntion, which in turn is enqueued in a SYCL queue.

## Microsoft's Visual Studio Code IDE (VSCode)
VSCode orgranises each and every individual C++ project with a set of settings and configuration files. These files are contained in a hidden directory named **.vscode**. The configuration files **tasks.json** and **launch.json** are used to configure the building (generally compiling) of the project followed by the debugging the program respectively. The other files like **settings.json** and **c_cpp_properties.json** are used by VSCode to inform it of your preferences and development environment. To learn more about how to use VSCode with DPC++, see the series of VSCode blogs at [here](https://codeplay.com/portal/blogs/2023/03/01/setting-up-c-development-with-visual-studio-code-on-ubuntu). Some paths defined in the configuration files are hardcoded to the author's system (see Development Environment below). While it is likely these paths will be incompatible with another's system, please review and edit the paths to fit with your current system.

### VSCode and git
As VSCode configuration files may have settings that are local to your development environment, you are likely to want to keep them, but not likely want them to be staged to be commited and so pushed up stream, nor have git overwrite your local settings.
For the case of not wanting to push local changes, the git .gitignore file handles this. For case where local settings are to be kept you may want to do the following: 
Perform a git stash, before getting the latest changes from the repository (by using git pull origin master or git rebase origin/master), and then merge your changes from the stash using git stash pop stash@{0}.

## VSCode project build instructions
Both the sub-projects require access to the NVIDIA helper header files in the NVIDIA cuda-samples-nvidia ```Common``` directory (which contained the original NVIDIA CUDA ```5_Domain_Specific/dxtc``` project). Ensure the CUDA samples **Common** directory is included in the makefile for the static library build. The INCLUDE_PATH is neccesary as the CUDA code pulls in various helper files, i.e., helper_functions.h, in order to compile successfully.

The same CUDA samples **Common** directory also needs to be included in the VSCode project **StaticLibrarySYCLUsesCUDAKernels** to build the SYCL code with the CUDA static library. Edit the line "-I${workspaceFolder}/< change path to >/cuda-samples-nvidia/Common" in each of the build configurations in the project's task.json file. Similarly, the VSCode ```.vscode/c_cpp_properties.json``` files for each project require the path to the helper files. For example:
```
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "${workspaceFolder}/../../cuda-samples-nvidia/Common",  <=== change this line
                "/usr/local/cuda-12.0/targets/x86_64-linux/include"
            ],
            "defines": [],
            "compilerPath": "/usr/local/cuda/bin/nvcc",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-clang-x64"
        }
    ],
    "version": 4
}
```

Next, build the static library which contains the CUDA kernels. 
Using VSCode, open the VSCode project folder *VSCodeStaticLibraryMakeCUDAKernesDXTC*. Select the **task.json** file in the **.vscode** directory and use key shift+alt+b to use the default build task. The build task uses the makefile to build the static library in the **bin** directory.

Once the static library is built, it is linked to the main program to enable the SYCL program to call upon the CUDA kernels. Open up a new project window using VSCode for the folder *VSCodeStaticLibraryUse*. Use shift+alt+b to select an **icpx** compiler build and hit return. The main project will build putting the executable in the bin directory.

## Configuring the build using CMake

For each sub-project, locate the CMakelist.txt in each and modify the CMake variable ```CUDA_TOOLKIT_HELPER_FILES``` to point to where you have installed the NVIDIA cuda-samples-nvidia ```Common``` directory (which contained the original NVIDIA CUDA ```5_Domain_Specific/dxtc``` project). This will provide access to and so include the necessary helper header files to enable a successful build.

To configure and build both the static CUDA kernel library and link to the application, do the following:

```
mkdir build
cd build
cmake ..
make
```

The configuration and resultant application executable ```CUDAInteropSYCLUSMSimple_d``` will be put in the CMake ```build``` directory build/StaticLibrarySYCLUsesCUDAKernels.

The application requires access to the image file ```teapot512_std.ppm``` to operate as expected. Once the CMake and make have completed successfully, copy the folder ```StaticLibrarySYCLUsesCUDAKernels/data``` to the CMake build directory. For example:
```
cd build/StaticLibrarySYCLUsesCUDAKernels
cp -r ../../StaticLibrarySYCLUsesCUDAKernels/data ..
``` 
The application can now be executed.

## Application results
Once the application has been successfully built, on its execution the output should be similar to:
```
./CUDAInteropSYCLUSMSimple_d Starting...
Running on device: NVIDIA GeForce RTX 2060

Using image: ././../data/teapot512_std.ppm 
Image Loaded '././../data/teapot512_std.ppm', w:512 x h:512 pixels

Running DXT Compression on 512 x 512 image...

16384 Blocks, 64 Threads per Block, 1048576 Threads in Grid...

dxtc, Throughput = 120.5814 MPixels/s, Time = 0.00217 s, Size = 262144 Pixels, NumDevsUsed = 1, Workgroup = 64

Checking accuracy...
RMS(reference, result) = 0.000000

Test passed
```

## Development environment
Ubuntu 22.04 LTS \
Intel oneAPI Base Toolkit 2023.1 + CUDA Plugin 2023.1 (setenvars.sh is active) \
NVIDIA CUDA Toolkit 12.0 \
NVIDIA CUDA Samples Common directory for CUDA helper include files (/cuda-samples-nvidia/Common) \
Compilers: NVIDIA's nvcc and Intel's icpx \
Target device: NVIDIA GTX 2060 \
Development IDE: Microsoft VSCode + extensions: Microsoft C/C++ pack \
CMake 3.22.1.
