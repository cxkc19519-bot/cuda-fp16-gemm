#include "common.cuh"
#include "gemm.h"

#include <cuda_fp16.h>

namespace cuda_gemm {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__global__ void gemm_coalesced_kernel(const __half* __restrict__ a,
                                      const __half* __restrict__ b,
                                      __half* __restrict__ c,
                                      int m,
                                      int n,
                                      int k) {
    // CUDA linearizes a 2D block with x as the fastest-moving dimension.
    // Mapping x to the row-major contiguous N dimension lets neighboring
    // lanes load B and store C at neighboring addresses.
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (row >= m || col >= n) {
        return;
    }

    float accumulator = 0.0F;
    for (int inner = 0; inner < k; ++inner) {
        const float a_value = __half2float(a[row * k + inner]);
        const float b_value = __half2float(b[inner * n + col]);
        accumulator = fmaf(a_value, b_value, accumulator);
    }

    c[row * n + col] = __float2half_rn(accumulator);
}

}  // namespace

void launch_gemm_coalesced(const __half* a,
                           const __half* b,
                           __half* c,
                           int m,
                           int n,
                           int k,
                           cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(ceil_div(n, kBlockX), ceil_div(m, kBlockY));
    gemm_coalesced_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_KERNEL_CHECK();
}

}  // namespace cuda_gemm

