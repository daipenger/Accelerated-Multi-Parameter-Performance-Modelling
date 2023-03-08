cmake_minimum_required(VERSION 3.1)
project(pmpp)

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=skylake -fopenmp")

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lcublas -gencode arch=compute_52,code=sm_52")

cuda_add_executable(pmpp
	PARAMETERS.cc
	main.cc
	matrix.cc
	LeastSquaresGPU.cc
	gpuKernels.cu)

cuda_add_library(pmpp-perf SHARED
	PARAMETERS.cc
	main.cc
	matrix.cc
    LeastSquaresGPU.cc
	gpuKernels.cu)

target_link_libraries(pmpp ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(pmpp-perf ${CUDA_CUBLAS_LIBRARIES})
