#include "common.cuh"
#include "gemm.h"

#include <cuda_fp16.h>

namespace cuda_gemm {
namespace {

template <int kTile>
__global__ void gemm_shared_kernel(const __half* __restrict__ a,
                                   const __half* __restrict__ b,
                                   __half* __restrict__ c,
                                   int m,
                                   int n,
                                   int k) {
    static_assert(kTile * kTile <= 1024, "CUDA thread block exceeds 1024 threads");
    __shared__ __half a_tile[kTile][kTile];
    __shared__ __half b_tile[kTile][kTile];

    const int local_col = static_cast<int>(threadIdx.x);
    const int local_row = static_cast<int>(threadIdx.y);
    const int row = static_cast<int>(blockIdx.y) * kTile + local_row;
    const int col = static_cast<int>(blockIdx.x) * kTile + local_col;

    float accumulator = 0.0F;
    const int tile_count = ceil_div(k, kTile);

    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_col = tile * kTile + local_col;
        const int b_row = tile * kTile + local_row;

        // Zero fill makes the same inner loop valid for partial edge tiles.
        a_tile[local_row][local_col] =
            (row < m && a_col < k) ? a[row * k + a_col] : __float2half_rn(0.0F);
        b_tile[local_row][local_col] =
            (b_row < k && col < n) ? b[b_row * n + col] : __float2half_rn(0.0F);

        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kTile; ++inner) {
            accumulator = fmaf(__half2float(a_tile[local_row][inner]),
                               __half2float(b_tile[inner][local_col]),
                               accumulator);
        }

        // Every thread must finish reading this stage before it is overwritten.
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = __float2half_rn(accumulator);
    }
}

template <int kTile>
void launch_gemm_shared_tile(const __half* a,
                             const __half* b,
                             __half* c,
                             int m,
                             int n,
                             int k,
                             cudaStream_t stream) {
    const dim3 block(kTile, kTile);
    const dim3 grid(ceil_div(n, kTile), ceil_div(m, kTile));
    gemm_shared_kernel<kTile><<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_KERNEL_CHECK();
}

}  // namespace

void launch_gemm_shared(const __half* a,
                        const __half* b,
                        __half* c,
                        int m,
                        int n,
                        int k,
                        cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    launch_gemm_shared_tile<32>(a, b, c, m, n, k, stream);
}

void launch_gemm_shared_16(const __half* a,
                           const __half* b,
                           __half* c,
                           int m,
                           int n,
                           int k,
                           cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    launch_gemm_shared_tile<16>(a, b, c, m, n, k, stream);
}

}  // namespace cuda_gemm
