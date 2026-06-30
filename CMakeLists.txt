cmake_minimum_required(VERSION 3.18)

project(cuda_sgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# let every target find headers in the include/
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- executables ---   
# device query, print the card spec
add_executable(device_query src/device_query.cu)

# correctness test - kernel vs CPU reference
# add_executable(correctness tests/correctness.cu src/kernels/naive.cu)

# benchmark - times the kernel, needs cuBLAS for comparision later
# add_executable(benchmark src/bench/benchmark.cu src/kernels/naive.cu)

# link cuBLAS into the benchmark ( for the cuBLAS comparision milestone)
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(benchmark PRIVATE CUDA::cublas)