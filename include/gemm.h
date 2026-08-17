#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cuda_gemm {

enum class GemmKernel {
    kMmaPtx,
    kWmmaAsync,
};

// All matrices use row-major storage:
//   A[M, K], B[K, N], C[M, N].
// Inputs and output are FP16; accumulation is FP32.
void launch_gemm_naive(const __half* a,
                       const __half* b,
                       __half* c,
                       int m,
                       int n,
                       int k,
                       cudaStream_t stream = nullptr);

void launch_gemm_coalesced(const __half* a,
                           const __half* b,
                           __half* c,
                           int m,
                           int n,
                           int k,
                           cudaStream_t stream = nullptr);

void launch_gemm_shared(const __half* a,
                        const __half* b,
                        __half* c,
                        int m,
                        int n,
                        int k,
                        cudaStream_t stream = nullptr);

// Experimental lower-occupancy-pressure variant used to compare 16x16 and
// 32x32 one-output-per-thread shared-memory tiles.
void launch_gemm_shared_16(const __half* a,
                           const __half* b,
                           __half* c,
                           int m,
                           int n,
                           int k,
                           cudaStream_t stream = nullptr);

void launch_gemm_register(const __half* a,
                          const __half* b,
                          __half* c,
                          int m,
                          int n,
                          int k,
                          cudaStream_t stream = nullptr);

void launch_gemm_register_fp32_smem(const __half* a,
                                    const __half* b,
                                    __half* c,
                                    int m,
                                    int n,
                                    int k,
                                    cudaStream_t stream = nullptr);

void launch_gemm_register_interleaved(const __half* a,
                                      const __half* b,
                                      __half* c,
                                      int m,
                                      int n,
                                      int k,
                                      cudaStream_t stream = nullptr);

void launch_gemm_register_vectorized(const __half* a,
                                     const __half* b,
                                     __half* c,
                                     int m,
                                     int n,
                                     int k,
                                     cudaStream_t stream = nullptr);

void launch_gemm_register_vectorized_store(const __half* a,
                                           const __half* b,
                                           __half* c,
                                           int m,
                                           int n,
                                           int k,
                                           cudaStream_t stream = nullptr);

void launch_gemm_wmma(const __half* a,
                      const __half* b,
                      __half* c,
                      int m,
                      int n,
                      int k,
                      cudaStream_t stream = nullptr);

void launch_gemm_wmma_block(const __half* a,
                            const __half* b,
                            __half* c,
                            int m,
                            int n,
                            int k,
                            cudaStream_t stream = nullptr);

void launch_gemm_wmma_block_64(const __half* a,
                               const __half* b,
                               __half* c,
                               int m,
                               int n,
                               int k,
                               cudaStream_t stream = nullptr);

void launch_gemm_wmma_async(const __half* a,
                            const __half* b,
                            __half* c,
                            int m,
                            int n,
                            int k,
                            cudaStream_t stream = nullptr);

void launch_gemm_mma_ptx(const __half* a,
                         const __half* b,
                         __half* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream = nullptr);

// Policy calibrated on RTX 3060 (sm_86) and RTX 4090 (sm_89) with K=N=4096
// LLM decode/prefill benchmarks. Keeping selection separate makes the policy
// testable without coupling it to the kernel implementations.
GemmKernel select_gemm_kernel(int m, int n, int k);

void launch_gemm_dispatch(const __half* a,
                          const __half* b,
                          __half* c,
                          int m,
                          int n,
                          int k,
                          cudaStream_t stream = nullptr);

// cuBLAS uses column-major storage by default. This wrapper computes
// C^T = B^T * A^T so callers can keep the same row-major contract.
void launch_cublas_gemm(cublasHandle_t handle,
                        const __half* a,
                        const __half* b,
                        __half* c,
                        int m,
                        int n,
                        int k);

}  // namespace cuda_gemm
