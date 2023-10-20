// SYCL or oneAPI toolkit headers:
#include <sycl/sycl.hpp>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// In-house headers:
#include "staticlibfnscudakernel.h"

// Third party headers:
#include <iostream>
#include <vector>
#include <iostream>
#include <string>

// Typedefs:
typedef int Data_t;
typedef std::vector< Data_t > VectorInt_t; 

// Forward declarations:
void StaticLibFnsCUDAKernel::HelloWorldFacade();
cudaError_t StaticLibFnsCUDAKernel::KernelCUDAVectorAddFacade( const int vBlocksPerGrid, const int vThreadsPerBlock, const int *vpPtrAccA, const int *vpPtrAccB, int *vpPtrAccSum, const int vNElements );
void HelloFromCUDAKernel( sycl::queue &vQ );
void VectorAdditionUsingCUDA( sycl::queue &vQ, const VectorInt_t &vVecA, const VectorInt_t &vVecB, VectorInt_t &vVecSumParallel );
void VectorAdditionUsingSYCL( sycl::queue &vQ, const VectorInt_t &vVecA, const VectorInt_t &vVecB, VectorInt_t &vVecSumParallel );

// Asynchronous errors hander, catch faults in asynchronously executed code
// inside a command group or a kernel. They can occur in a different stackframe, 
// asynchronous error cannot be propagated up the stack. 
// By default, they are considered 'lost'. The way in which we can retrieve them
// is by providing an error handler function.
auto exception_handler = []( sycl::exception_list vExceptions ) 
{
    for( std::exception_ptr const &e : vExceptions ) 
    {
        try 
        {
          std::rethrow_exception( e );
        } 
        catch( sycl::exception const &e ) 
        {
          std::cout << "Queue handler caught asynchronous SYCL exception:\n" << e.what() << std::endl;
        }
    }
};

// Defining a SYCL kernel as function object to carry out a vector add
class KernelSYCLVectorAdd
{
  public:
    using AccessorRead_t = sycl::accessor< Data_t, 1, sycl::access::mode::read, sycl::access::target::device >;
    using AccessorWrite_t = sycl::accessor< Data_t, 1, sycl::access::mode::write, sycl::access::target::device >;
        
    KernelSYCLVectorAdd( AccessorRead_t vAccA, AccessorRead_t vAccB, AccessorWrite_t vAccSum )
    : m_accA( vAccA )
    , m_accB( vAccB )
    , m_accSum ( vAccSum )
    {}

    // The parameter of the operator is the work item id.
    void operator()( sycl::id< 1 > vI ) const 
    { 
        m_accSum[ vI ] = m_accA[ vI ] + m_accB[ vI ]; 
    }

  private:
    // Create an accessor for each buffer with access permission: read, write or
    // read/write. The accessor is a mean to access the memory in the buffer.
    AccessorRead_t  m_accA;
    AccessorRead_t  m_accB;
    AccessorWrite_t m_accSum;
};
    
// The program that enqueues CUDA kernels alongside SYCL kernels
int main( void )
{
    std::cout << "Hello from this program." << std::endl;
    
    constexpr Data_t vecSize = 10000;
    VectorInt_t vecA( vecSize, 1 );
    VectorInt_t vecB( vecSize, 1 );
    VectorInt_t vecSumParallel( vecSize, 0 );
    
    try
    {
        const auto deviceSelector = sycl::default_selector_v;
        sycl::queue q( deviceSelector, exception_handler );
        
        std::cout << "Running on device: " << q.get_device().get_info< sycl::info::device::name >() << "\n";

        // Note the following kernels are not necessarily executed in the order show here.
        // Kernels can execute out of order (asynchronous) if the call graph does not see any 
        // dependency on other kernels' outputs or unless specifically synced.
        HelloFromCUDAKernel( q );
        VectorAdditionUsingSYCL( q, vecA, vecB, vecSumParallel );
        VectorAdditionUsingCUDA( q, vecSumParallel, vecB, vecA );
        VectorAdditionUsingSYCL( q, vecA, vecB, vecSumParallel );
    }
    catch( std::exception const &e ) 
    {
        std::cout << "An exception is caught while computing on device.\n";
        std::terminate();
    }

    // Print out the result of vector add
    const int indices[] { 0, 1, 2, 3, vecSize - 1 };
    constexpr size_t indicesSize = sizeof( indices ) / sizeof( Data_t );
    for( int i = 0; i < indicesSize; i++ ) 
    { 
        const int j = indices[ i ];
        if( i == indicesSize - 1 ) std::cout << "...\n";
        std::cout << "[" << j << "]: " << vecA[j] << " + " << vecB[j] << " = " << vecSumParallel[j] << "\n";
    }

    // Tidy up - release resources
    vecA.clear();
    vecB.clear();
    vecSumParallel.clear();
    
    // Confirm all want well
    std::cout << "CUDA kernels interop with SYCL completed on device." << std::endl;
    
    return EXIT_SUCCESS;
}

// Function to enqueue a SYCL kernel to continue performing further additions
// on an existing vector of ints to an existing SYCL queue.
void VectorAdditionUsingSYCL( sycl::queue &vQ, const VectorInt_t &vVecA, const VectorInt_t &vVecB, VectorInt_t &vVecSumParallel )
{
    std::cout << "Hello, about to execute VectorAdditionUsingSYCL()\n";

    // Create the range object for the vectors managed by the buffer.
    const sycl::range< 1 > rngVecItems( vVecA.size() );

    // Create buffers that hold the data shared between the host and the devices.
    // The buffer destructor is responsible to copy the data back to host when it
    // goes out of scope.
    sycl::buffer bufA( vVecA.data(), rngVecItems );
    sycl::buffer bufB( vVecB.data(), rngVecItems );
    sycl::buffer bufSumParallel( vVecSumParallel.data(), rngVecItems );

    // Submit a command group to the queue by a C++ function that contains the
    // data access permission and device computation (kernel).
    vQ.submit( [&]( sycl::handler &vH ) 
    { 
        // Create an accessor for each buffer with access permission: read, write or
        // read/write. The accessor is a means to access the memory in the buffer.
        KernelSYCLVectorAdd::AccessorRead_t accA = bufA.get_access< sycl::access::mode::read>( vH );
        KernelSYCLVectorAdd::AccessorRead_t accB = bufB.get_access< sycl::access::mode::read>( vH );
        KernelSYCLVectorAdd::AccessorWrite_t accSum = bufSumParallel.get_access< sycl::access::mode::write >( vH );
        
        // Use parallel_for to run vector addition in parallel on device. This
        // executes the kernel.
        //    1st parameter is the number of work items.
        //    2nd parameter is the kernel, a functor class that specifies what to do per work item. 
        // By default SYCL uses unnamed kernels. This example is using a nameed lamda 'KernelSYCLVectorAdd'.
        vH.parallel_for( rngVecItems, KernelSYCLVectorAdd( accA, accB, accSum ) );
    });

    // Must be sure all is finished before calling the host_task()'s interop function
    vQ.wait_and_throw(); 
}

// Function to enqueue a CUDA kernel on to an existing SYCL queue
void HelloFromCUDAKernel( sycl::queue &vQ )
{
    vQ.submit( [&]( sycl::handler &vH ) 
    { 
        vH.host_task( [=]( /* No interop_handle needs to passed in if the backend interoperability is not required */ ) 
        {
            StaticLibFnsCUDAKernel::HelloWorldFacade();
        });
    });
}
        
// Function to enqueue a CUDA kernel to continue performing further additions
// on an existing vector of ints to an existing SYCL queue.
void VectorAdditionUsingCUDA( sycl::queue &vQ, const VectorInt_t &vVecA, const VectorInt_t &vVecB, VectorInt_t &vVecSumParallel )
{
    std::cout << "Hello, about to execute VectorAdditionUsingCUDA()\n";

    const int nElements = vVecA.size();

    // Create the range object for the vectors managed by the buffer.
    const sycl::range< 1 > rngVecItems( nElements );

    // Create buffers that hold the data shared between the host and the devices.
    // The buffer destructor is responsible to copy the data back to host when it
    // goes out of scope.
    sycl::buffer bufA( vVecA.data(), rngVecItems );
    sycl::buffer bufB( vVecB.data(), rngVecItems );
    sycl::buffer bufSumParallel( vVecSumParallel.data(), rngVecItems );
    
    // Submit a command group to the queue by a C++ function that contains the
    // data access permission and device computation (kernel).
    vQ.submit( [&]( sycl::handler &vH ) 
    { 
        sycl::accessor accA( bufA, vH, sycl::read_only );
        sycl::accessor accB( bufB, vH, sycl::read_only );
        sycl::accessor accSum( bufSumParallel, vH, sycl::write_only );
    
        vH.host_task( [=]( const sycl::interop_handle &vIh ) 
        {
            // Error code to check return values for CUDA calls
            cudaError_t err = cudaSuccess;

            // Make compatible CUDA device pointers
            const CUdeviceptr accA_ptr = vIh.get_native_mem< sycl::backend::ext_oneapi_cuda >( accA );
            const CUdeviceptr accB_ptr = vIh.get_native_mem< sycl::backend::ext_oneapi_cuda >( accB );
            CUdeviceptr accSum_ptr = vIh.get_native_mem< sycl::backend::ext_oneapi_cuda >( accSum );
            const Data_t *rawHostPtrAccA = reinterpret_cast< const Data_t * >( accA_ptr );
            const Data_t *rawHostPtrAccB = reinterpret_cast< const Data_t * >( accB_ptr );
            Data_t *rawHostPtrAccSum = reinterpret_cast< Data_t * >( accSum_ptr );
            
            // CUDA number of threads in each thread block
            const int threadsPerBlock = 1024;
            // CUDA number of thread blocks in grid
            const int blocksPerGrid = (nElements + threadsPerBlock - 1) / threadsPerBlock;
            // Call the CUDA kernel directly from SYCL. Pointers are to device data.
            err = StaticLibFnsCUDAKernel::KernelCUDAVectorAddFacade( blocksPerGrid, threadsPerBlock, 
                rawHostPtrAccA, rawHostPtrAccB, rawHostPtrAccSum, vVecA.size() );
            if( err != cudaSuccess )  
            { 
                std::cerr << "Failed to launch KernelCUDAVectorAddFacade (error code " << cudaGetErrorString( err ) << ")!\n";
            }           
        });
    });
}
