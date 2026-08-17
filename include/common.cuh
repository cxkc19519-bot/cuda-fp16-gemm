#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace cuda_gemm {

template <typename T>
__host__ __device__ constexpr T ceil_div(T value, T divisor) {
    return (value + divisor - 1) / divisor;
}

inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::cerr << "CUDA error at " << file << ':' << line << " for " << expression
              << ": " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
}

inline const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline void check_cublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }

    std::cerr << "cuBLAS error at " << file << ':' << line << " for " << expression
              << ": " << cublas_status_string(status) << '\n';
    std::exit(EXIT_FAILURE);
}

}  // namespace cuda_gemm

#define CUDA_CHECK(expression) \
    ::cuda_gemm::check_cuda((expression), #expression, __FILE__, __LINE__)

#define CUBLAS_CHECK(expression) \
    ::cuda_gemm::check_cublas((expression), #expression, __FILE__, __LINE__)

#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
