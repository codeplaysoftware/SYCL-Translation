
## [main]
if(NOT DEFINED ${CMAKE_BUILD_TYPE})
	set(CMAKE_BUILD_TYPE "Debug")
endif()

# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies.
cmake_minimum_required(VERSION 3.11...3.27)

message(STATUS ">>SYCLBufferExecCUDAKernels build started")

# set the project name
project(SYCLBufferExecCUDAKernels
        VERSION 1.0
        DESCRIPTION "SYCL Buffer model program executing CUDA kernels"
        LANGUAGES CXX)

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)
## [start set variables]
# Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(TARGET_NAME CUDAInteropSYCLSimple)
set(STATIC_LIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/StaticLibraryMakeCUDAKernel/src")
set(STATIC_LIB_DIR "${CMAKE_SOURCE_DIR}/StaticLibraryMakeCUDAKernel/bin")
set(CMAKE_CXX_COMPILER icpx)
set(COMPILE_FLAGS "-fsycl")
string(APPEND COMPILE_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda")
string(APPEND COMPILE_FLAGS " -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_75")
string(APPEND COMPILE_FLAGS " -fdiagnostics-color=always")
string(APPEND COMPILE_FLAGS " -I${STATIC_LIB_INCLUDE_DIR}")
string(APPEND COMPILE_FLAGS " -I/usr/local/cuda/include")
set(COMPILE_LINK_FLAGS "-fsycl")
string(APPEND COMPILE_LINK_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda")
string(APPEND COMPILE_LINK_FLAGS " -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_75")
string(APPEND COMPILE_LINK_FLAGS " -L/usr/local/cuda/lib64")
string(APPEND COMPILE_LINK_FLAGS " -lcudart")
set(SOURCE_FILE src/main.cpp)
## [end set variables]

# Find packages go here
## [find_package]
## [find_package]

# Sets up global settings here

# Add other deeper CMakeList sub directories

# Adding an executable program and linking to the static library
# To compile in a single command:
#    icpx -fsycl <file>.cpp -o <file>
# CMake executes:
#    [compile] icpx -fsycl -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl <file>.cpp.o -o <file>
## [add_and_link]
add_library(StaticLibCudaKernels SHARED IMPORTED)
add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${COMPILE_LINK_FLAGS}")
target_link_libraries(${TARGET_NAME} StaticLibOfFnsCUDAKernelcmake_d)
add_dependencies(${TARGET_NAME} StaticLibCudaKernels)
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})
## [add_and_link]

## [main]
message(STATUS "<<SYCLBufferExecCUDAKernels build finished")