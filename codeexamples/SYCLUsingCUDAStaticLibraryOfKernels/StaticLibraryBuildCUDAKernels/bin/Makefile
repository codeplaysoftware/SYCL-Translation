# Library names
SRC_FILE ?= library.cu
DST_FILE ?= libStaticLibOfFnsCUDAKernel

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda/bin

INCLUDE_PATH ?= <change this path to>/cuda-samples-nvidia/Common

# Target rules
all: build_debug

build_debug:
	$(CUDA_PATH)/nvcc -rdc=true -c -ccbin g++ -I$(INCLUDE_PATH) -I/usr/local/cuda/bin/../targets/x86_64-linux/include -m64 --threads 0 --std=c++11 -gencode arch=compute_75,code=sm_75 -g -G -O0 ../src/$(SRC_FILE) -o $(DST_FILE)_d_tmp.o 
	$(CUDA_PATH)/nvcc -dlink -o  libStaticLibOfFnsCUDAKernel_d.o  libStaticLibOfFnsCUDAKernel_d_tmp.o -lcudart
	ar cru  $(DST_FILE)_d.a  $(DST_FILE)_d.o $(DST_FILE)_d_tmp.o
	ranlib $(DST_FILE)_d.a

build:
	$(CUDA_PATH)/nvcc -rdc=true -c -ccbin g++ -I$(INCLUDE_PATH) -I/usr/local/cuda/bin/../targets/x86_64-linux/include -m64 --std=c++11 -gencode arch=compute_75,code=sm_75 ../src/$(SRC_FILE) -o $(DST_FILE)_tmp.o 
	$(CUDA_PATH)/nvcc -dlink -o  libStaticLibOfFnsCUDAKernel.o  libStaticLibOfFnsCUDAKernel_tmp.o -lcudart
	ar cru  $(DST_FILE).a  $(DST_FILE).o $(DST_FILE)_tmp.o
	ranlib $(DST_FILE).a

clean:
	rm *.o
	rm *.a
	
