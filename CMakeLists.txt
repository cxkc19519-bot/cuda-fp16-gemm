cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # RTX 3060 = sm_86. Override with -DCMAKE_CUDA_ARCHITECTURES=89 for RTX 4090.
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build" FORCE)
endif()

project(cuda_fp16_gemm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(GEMM_BUILD_TESTS "Build correctness tests" ON)
option(GEMM_BUILD_BENCHMARKS "Build benchmark executable" ON)

if(MSVC)
    # NVIDIA headers contain characters outside the active Simplified Chinese
    # code page. Suppress this third-party-header warning for CUDA compilation.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819,/wd4211>)
endif()

add_library(cuda_gemm STATIC
    src/gemm_naive.cu
    src/gemm_coalesced.cu
    src/gemm_shared.cu
    src/gemm_register.cu
    src/gemm_wmma.cu
    src/gemm_mma_ptx.cu
    src/dispatcher.cu
    benchmark/cublas_gemm.cu
)

target_include_directories(cuda_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cuda_gemm PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(cuda_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(cuda_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

if(MSVC)
    target_compile_options(cuda_gemm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
else()
    target_compile_options(cuda_gemm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
endif()

if(GEMM_BUILD_TESTS)
    enable_testing()
    add_executable(test_correctness tests/test_correctness.cu)
    target_link_libraries(test_correctness PRIVATE cuda_gemm)
    add_test(NAME gemm_correctness COMMAND test_correctness)
endif()

if(GEMM_BUILD_BENCHMARKS)
    add_executable(gemm_benchmark benchmark/benchmark.cu)
    target_link_libraries(gemm_benchmark PRIVATE cuda_gemm)
endif()
