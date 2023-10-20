# SYCL Buffer model program executing CUDA kernels

This project shows how to mix CUDA kernels with SYCL kernels in the same SYCL code base using one task graph. It performs a simple Vector add where the data values are incremented by first a SYCL kernel, then a CUDA kernel, followed by a SYCL kernel.

## CUDA interopability approach with SYCL
All the CUDA kernels are put in a static library. Each kernel is wrapped by a C++ Facade function. This library is compiled with the NVIDIA compiler.

The static library is linked to a SYCL project. The Facade wrapper functions are called from within a SYCL host_task() fucntion, which in turn is enqueued in a SYCL queue along with other SYCL kernels.

## Microsoft's Visual Studio Code IDE (VSCode)
VSCode orgranises each and every individual C++ project with a set of settings and configuration files. These files are contained in a hidden directory named **.vscode**. The configuration files **tasks.json** and **launch.json** are used to configure the building (generally compiling) of the project  followed by the debugging the program respectively. The other files like **settings.json** and **c_cpp_properties.json** are used by VSCode to inform it of your preferences and development environment. To learn more about how to use VSCode with DPC++, see the series of VSCode blogs at [here](https://codeplay.com/portal/blogs/2023/03/01/setting-up-c-development-with-visual-studio-code-on-ubuntu).  Some paths defined in the configuration files are hardcoded to the author's system (see Development Environment below). While it is likely these paths will be incompatible with another's system, please review and edit the paths to fit with your current system.

### VSCode and git
As VSCode configuration files may have settings that are local to your development environment, you are likely to want to keep them, but not likely want them to be staged to be commited and so pushed up stream, nor have git overwrite your local settings.
For the case of not wanting to push local changes, the git .gitignore file handles this. For case where local settings are to be kept you may want to do the following: 
Perform a git stash, before getting the latest changes from the repository (by using git pull origin master or git rebase origin/master), and then merge your changes from the stash using git stash pop stash@{0}.

## Build instructions
### Using VSCode
First we need to buid the static library which contains the CUDA kernels. 
Using Microsoft's Visual Studio Code IDE (VSCode), open the VSCode project folder *StaticLibraryMakeCUDAKernel*. Select the **task.json** file in the **.vscode** directory and use key shift+alt+b to use the default build task. The build task uses the makefile to build the static library in the **bin** directory.

Once the static library is built, it is linked to the main program to enable the SYCL program to call upon the CUDA kernels. Open up a new project window using VSCode for the folder *StaticLibraryUse*. Use shift+alt+b to select an **icpx** compiler build and hit return. The main project will build putting the executable in the bin directory.

### Configuring using CMake
To configure and build both the static CUDA kernel library and link to the application, invoke CMake as follows:

```
mkdir build

cd build

cmake ..
```
The configuration and resultant application executable will be put in the CMAKE build directory StaticLibraryUse.

## Development environment
Ubuntu 22.04 LTS \
Intel oneAPI Base Toolkit 2023.1 + CUDA Plugin 2023.1 (setenvars.sh is active) \
NVIDIA CUDA Toolkit 12.0 \
NVIDIA CUDA Samples Common directory for CUDA helper include files (/cuda-samples-nvidia/Common) \
Compilers: NVIDIA's nvcc and Intel's icpx \
Target device: NVIDIA GTX2060 \
Development IDE: Microsoft VSCode v1.82 + extensions: Microsoft C/C++ pack \
CMake version 3.22.1.
