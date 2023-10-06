
## [main]

# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies.
cmake_minimum_required(VERSION 3.11...3.27)

message(STATUS ">Top level CMakeList started")

# set the project name
project(ProjectSYCLBufferExecCUDAKernels)

# Set CMake input options' default values here (if any used)

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)
## [set variables]
# Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_BUILD_TYPE "Debug")
## [set variables]

# Find packages go here
## [find_package]
## [find_package]

# Sets up global settings here

# Add the CUDA kernels library + other sub directories
add_subdirectory(StaticLibraryMakeCUDAKernel)
add_subdirectory(StaticLibraryUse)

# Adding an executable program and linking to the static library
## [add_and_link]
## [add_and_link]

## [main]
message(STATUS "<Top level CMakeList finished")
