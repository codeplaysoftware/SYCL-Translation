
## [main]
if(NOT DEFINED ${CMAKE_BUILD_TYPE})
	set(CMAKE_BUILD_TYPE "Debug")
endif()

# Almost all CMake files should start with this
# You should always specify a range with the newest
# and oldest tested versions of CMake. This will ensure
# you pick up the best policies. CUDA started at v3.8.
cmake_minimum_required(VERSION 3.8...3.27)

message(STATUS ">>Static library CUDA kernels build started")

# set the project name
project(StaticLibraryMakeCUDAKernel
        VERSION 1.0
        DESCRIPTION "Static library holding CUDA kernels"
        LANGUAGES CXX CUDA)

# If you set any CMAKE_ variables, that can go here.
# (But usually don't do this, except maybe for C++ standard)
## [start set variables]
# Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(TARGET_NAME StaticLibOfFnsCUDAKernelcmake_d)
set(STATIC_LIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/StaticLibraryMakeCUDAKernel/src")
set(STATIC_LIB_FILES "${STATIC_LIB_INCLUDE_DIR}/library.cu")
## [end set variables]

# Find packages go here
## [find_package]
# The CUDA package has been deprecated and will cause errors when using CMake 3.27 or greater
#find_package(CUDA REQUIRED)
## [find_package]

# Sets up global settings here

# Add other deeper CMakeList sub directories

# Create the static library
## [add_and_link]
add_library(${TARGET_NAME} SHARED ${STATIC_LIB_FILES})
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_11)
# We need to explicitly state that we need all CUDA files in this
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# Enables both relocatable device code compiling and linking:
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "75")
# Redirect binaries to the project folder
#set_target_properties(${TARGET_NAME} PROPERTIES
#    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_HOME}"
#    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_HOME}"
#    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_HOME}"
target_link_libraries(${TARGET_NAME} PUBLIC ${CMAKE_THREAD_LIBS_INIT}  ${CMAKE_DL_LIBS})
## [add_and_link]

## [main]
message(STATUS "<<Static library CUDA kernels build finished")
