#include "common.cuh"
#include "gemm.h"

#include <cuda_fp16.h>

#include <type_traits>

namespace cuda_gemm {
namespace {

constexpr int kBlockTileM = 64;
constexpr int kBlockTileN = 64;
constexpr int kBlockTileK = 16;
constexpr int kThreadTileM = 4;
constexpr int kThreadTileN = 4;
constexpr int kThreadsX = kBlockTileN / kThreadTileN;
constexpr int kThreadsY = kBlockTileM / kThreadTileM;
constexpr int kThreadCount = kThreadsX * kThreadsY;
constexpr int kTransposedAStride = kBlockTileM + 2;

static_assert(kBlockTileM % kThreadTileM == 0);
static_assert(kBlockTileN % kThreadTileN == 0);
static_assert(kThreadCount == 256);

template <typename SharedType>
__device__ __forceinline__ SharedType to_shared(__half value) {
    if constexpr (std::is_same_v<SharedType, float>) {
        return __half2float(value);
    } else {
        return value;
    }
}

template <typename SharedType>
__device__ __forceinline__ float from_shared(SharedType value) {
    if constexpr (std::is_same_v<SharedType, float>) {
        return value;
    } else {
        return __half2float(value);
    }
}

template <typename SharedType,
          bool kInterleavedColumns,
          bool kVectorizedShared,
          bool kVectorizedStore>
__global__ void gemm_register_kernel(const __half* __restrict__ a,
                                     const __half* __restrict__ b,
                                     __half* __restrict__ c,
                                     int m,
                                     int n,
                                     int k) {
    __shared__ SharedType a_tile[kVectorizedShared
                                     ? kBlockTileK * kTransposedAStride
                                     : kBlockTileM * kBlockTileK];
    __shared__ SharedType b_tile[kBlockTileK * kBlockTileN];

    static_assert(!kVectorizedShared || std::is_same_v<SharedType, __half>);
    static_assert(!kVectorizedStore || !kInterleavedColumns);

    const int thread_col = static_cast<int>(threadIdx.x);
    const int thread_row = static_cast<int>(threadIdx.y);
    const int thread_id = thread_row * kThreadsX + thread_col;
    const int block_row = static_cast<int>(blockIdx.y) * kBlockTileM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockTileN;
    const int output_row = block_row + thread_row * kThreadTileM;
    const int output_col = block_col +
                           (kInterleavedColumns ? thread_col
                                                : thread_col * kThreadTileN);

    float accumulators[kThreadTileM][kThreadTileN] = {};
    float a_registers[kThreadTileM];
    float b_registers[kThreadTileN];

    const int tile_count = ceil_div(k, kBlockTileK);
    for (int tile = 0; tile < tile_count; ++tile) {
        const int k_base = tile * kBlockTileK;

        // Cooperatively load the 64x16 A tile. Each of 256 threads loads
        // four FP16 values, with neighboring lanes reading neighboring data.
        for (int index = thread_id; index < kBlockTileM * kBlockTileK;
             index += kThreadCount) {
            const int local_row = index / kBlockTileK;
            const int local_col = index % kBlockTileK;
            const int global_row = block_row + local_row;
            const int global_col = k_base + local_col;
            const int shared_index = kVectorizedShared
                                         ? local_col * kTransposedAStride + local_row
                                         : local_row * kBlockTileK + local_col;
            a_tile[shared_index] =
                (global_row < m && global_col < k)
                    ? to_shared<SharedType>(a[global_row * k + global_col])
                    : to_shared<SharedType>(__float2half_rn(0.0F));
        }

        // Cooperatively load the 16x64 B tile. Each thread also loads four
        // values, and the row-major N dimension is contiguous across lanes.
        for (int index = thread_id; index < kBlockTileK * kBlockTileN;
             index += kThreadCount) {
            const int local_row = index / kBlockTileN;
            const int local_col = index % kBlockTileN;
            const int global_row = k_base + local_row;
            const int global_col = block_col + local_col;
            b_tile[local_row * kBlockTileN + local_col] =
                (global_row < k && global_col < n)
                    ? to_shared<SharedType>(b[global_row * n + global_col])
                    : to_shared<SharedType>(__float2half_rn(0.0F));
        }

        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < kBlockTileK; ++inner) {
            if constexpr (kVectorizedShared) {
                const int a_base = inner * kTransposedAStride +
                                   thread_row * kThreadTileM;
                const int b_base = inner * kBlockTileN +
                                   thread_col * kThreadTileN;
                const __half2 a01 =
                    *reinterpret_cast<const __half2*>(&a_tile[a_base]);
                const __half2 a23 =
                    *reinterpret_cast<const __half2*>(&a_tile[a_base + 2]);
                const __half2 b01 =
                    *reinterpret_cast<const __half2*>(&b_tile[b_base]);
                const __half2 b23 =
                    *reinterpret_cast<const __half2*>(&b_tile[b_base + 2]);
                const float2 af01 = __half22float2(a01);
                const float2 af23 = __half22float2(a23);
                const float2 bf01 = __half22float2(b01);
                const float2 bf23 = __half22float2(b23);
                a_registers[0] = af01.x;
                a_registers[1] = af01.y;
                a_registers[2] = af23.x;
                a_registers[3] = af23.y;
                b_registers[0] = bf01.x;
                b_registers[1] = bf01.y;
                b_registers[2] = bf23.x;
                b_registers[3] = bf23.y;
            } else {
#pragma unroll
                for (int row = 0; row < kThreadTileM; ++row) {
                    a_registers[row] = from_shared<SharedType>(
                        a_tile[(thread_row * kThreadTileM + row) *
                                   kBlockTileK + inner]);
                }
#pragma unroll
                for (int col = 0; col < kThreadTileN; ++col) {
                    const int shared_col = kInterleavedColumns
                                               ? col * kThreadsX + thread_col
                                               : thread_col * kThreadTileN + col;
                    b_registers[col] = from_shared<SharedType>(
                        b_tile[inner * kBlockTileN + shared_col]);
                }
            }

#pragma unroll
            for (int row = 0; row < kThreadTileM; ++row) {
#pragma unroll
                for (int col = 0; col < kThreadTileN; ++col) {
                    accumulators[row][col] =
                        fmaf(a_registers[row], b_registers[col], accumulators[row][col]);
                }
            }
        }

        __syncthreads();
    }

    if constexpr (kVectorizedStore) {
#pragma unroll
        for (int row = 0; row < kThreadTileM; ++row) {
            const int global_row = output_row + row;
            const std::size_t output_index =
                static_cast<std::size_t>(global_row) * n + output_col;
            if (global_row < m && output_col + kThreadTileN <= n &&
                (output_index & 3U) == 0U) {
                const __half h0 = __float2half_rn(accumulators[row][0]);
                const __half h1 = __float2half_rn(accumulators[row][1]);
                const __half h2 = __float2half_rn(accumulators[row][2]);
                const __half h3 = __float2half_rn(accumulators[row][3]);
                const uint2 packed{
                    static_cast<unsigned int>(__half_as_ushort(h0)) |
                        (static_cast<unsigned int>(__half_as_ushort(h1)) << 16U),
                    static_cast<unsigned int>(__half_as_ushort(h2)) |
                        (static_cast<unsigned int>(__half_as_ushort(h3)) << 16U)};
                *reinterpret_cast<uint2*>(&c[output_index]) = packed;
            } else if (global_row < m) {
#pragma unroll
                for (int col = 0; col < kThreadTileN; ++col) {
                    if (output_col + col < n) {
                        c[output_index + col] =
                            __float2half_rn(accumulators[row][col]);
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int row = 0; row < kThreadTileM; ++row) {
#pragma unroll
            for (int col = 0; col < kThreadTileN; ++col) {
                const int global_row = output_row + row;
                const int global_col = output_col +
                                       (kInterleavedColumns ? col * kThreadsX : col);
                if (global_row < m && global_col < n) {
                    c[global_row * n + global_col] =
                        __float2half_rn(accumulators[row][col]);
                }
            }
        }
    }
}

template <typename SharedType,
          bool kInterleavedColumns,
          bool kVectorizedShared,
          bool kVectorizedStore>
void launch_gemm_register_typed(const __half* a,
                                const __half* b,
                                __half* c,
                                int m,
                                int n,
                                int k,
                                cudaStream_t stream) {
    const dim3 block(kThreadsX, kThreadsY);
    const dim3 grid(ceil_div(n, kBlockTileN), ceil_div(m, kBlockTileM));
    gemm_register_kernel<SharedType,
                         kInterleavedColumns,
                         kVectorizedShared,
                         kVectorizedStore>
        <<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_KERNEL_CHECK();
}

}  // namespace

void launch_gemm_register(const __half* a,
                          const __half* b,
                          __half* c,
                          int m,
                          int n,
                          int k,
                          cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    launch_gemm_register_typed<__half, false, false, false>(a, b, c, m, n, k, stream);
}

void launch_gemm_register_fp32_smem(const __half* a,
                                    const __half* b,
                                    __half* c,
                                    int m,
                                    int n,
                                    int k,
                                    cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    launch_gemm_register_typed<float, false, false, false>(a, b, c, m, n, k, stream);
}

void launch_gemm_register_interleaved(const __half* a,
                                      const __half* b,
                                      __half* c,
                                      int m,
                                      int n,
                                      int k,
                                      cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    launch_gemm_register_typed<__half, true, false, false>(a, b, c, m, n, k, stream);
}

void launch_gemm_register_vectorized(const __half* a,
                                     const __half* b,
                                     __half* c,
                                     int m,
                                     int n,
                                     int k,
                                     cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    launch_gemm_register_typed<__half, false, true, false>(a, b, c, m, n, k, stream);
}

void launch_gemm_register_vectorized_store(const __half* a,
                                           const __half* b,
                                           __half* c,
                                           int m,
                                           int n,
                                           int k,
                                           cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    launch_gemm_register_typed<__half, false, true, true>(
        a, b, c, m, n, k, stream);
}

}  // namespace cuda_gemm
