#include "common.cuh"
#include "gemm.h"

#include <cuda_fp16.h>
#include <mma.h>

namespace cuda_gemm {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpSize = 32;
constexpr int kBlockTileM = 64;
constexpr int kBlockTileN = 32;
constexpr int kWarpsM = kBlockTileM / kWmmaM;
constexpr int kWarpsN = kBlockTileN / kWmmaN;
constexpr int kWarpsPerBlock = kWarpsM * kWarpsN;

// Teaching baseline: one warp computes one 16x16 output tile. A and B are
// staged through shared memory so the same kernel also supports partial M/N/K
// tiles by zero-padding out-of-range values.
__global__ void gemm_wmma_kernel(const __half* __restrict__ a,
                                 const __half* __restrict__ b,
                                 __half* __restrict__ c,
                                 int m,
                                 int n,
                                 int k) {
    __shared__ __align__(32) __half a_tile[kWmmaM * kWmmaK];
    __shared__ __align__(32) __half b_tile[kWmmaK * kWmmaN];
    __shared__ __align__(32) float c_tile[kWmmaM * kWmmaN];

    const int lane = static_cast<int>(threadIdx.x);
    const int block_row = static_cast<int>(blockIdx.y) * kWmmaM;
    const int block_col = static_cast<int>(blockIdx.x) * kWmmaN;

    wmma::fragment<wmma::matrix_a,
                   kWmmaM,
                   kWmmaN,
                   kWmmaK,
                   __half,
                   wmma::row_major>
        a_fragment;
    wmma::fragment<wmma::matrix_b,
                   kWmmaM,
                   kWmmaN,
                   kWmmaK,
                   __half,
                   wmma::row_major>
        b_fragment;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0F);

    const int tile_count = ceil_div(k, kWmmaK);
    for (int tile = 0; tile < tile_count; ++tile) {
        const int k_base = tile * kWmmaK;

        for (int index = lane; index < kWmmaM * kWmmaK;
             index += kWarpSize) {
            const int local_row = index / kWmmaK;
            const int local_col = index % kWmmaK;
            const int global_row = block_row + local_row;
            const int global_col = k_base + local_col;
            a_tile[index] =
                (global_row < m && global_col < k)
                    ? a[static_cast<std::size_t>(global_row) * k + global_col]
                    : __float2half_rn(0.0F);
        }

        for (int index = lane; index < kWmmaK * kWmmaN;
             index += kWarpSize) {
            const int local_row = index / kWmmaN;
            const int local_col = index % kWmmaN;
            const int global_row = k_base + local_row;
            const int global_col = block_col + local_col;
            b_tile[index] =
                (global_row < k && global_col < n)
                    ? b[static_cast<std::size_t>(global_row) * n + global_col]
                    : __float2half_rn(0.0F);
        }

        __syncthreads();
        wmma::load_matrix_sync(a_fragment, a_tile, kWmmaK);
        wmma::load_matrix_sync(b_fragment, b_tile, kWmmaN);
        wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        __syncthreads();
    }

    wmma::store_matrix_sync(c_tile, accumulator, kWmmaN, wmma::mem_row_major);
    __syncthreads();

    for (int index = lane; index < kWmmaM * kWmmaN; index += kWarpSize) {
        const int local_row = index / kWmmaN;
        const int local_col = index % kWmmaN;
        const int global_row = block_row + local_row;
        const int global_col = block_col + local_col;
        if (global_row < m && global_col < n) {
            c[static_cast<std::size_t>(global_row) * n + global_col] =
                __float2half_rn(c_tile[index]);
        }
    }
}

template <int CopyBytes>
__device__ __forceinline__ void cp_async_copy(void* shared_destination,
                                             const void* global_source,
                                             int valid_bytes) {
    const unsigned int shared_address =
        static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
    if constexpr (CopyBytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                         "r"(shared_address),
                         "l"(global_source),
                         "r"(valid_bytes));
    } else {
        static_assert(CopyBytes == 4);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::
                         "r"(shared_address),
                         "l"(global_source),
                         "r"(valid_bytes));
    }
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// Eight warps cooperate on a 64x32 block tile. Each A tile is reused by two
// warps along N, and each B tile is reused by four warps along M.
template <int BlockN>
__global__ void gemm_wmma_block_kernel(const __half* __restrict__ a,
                                       const __half* __restrict__ b,
                                       __half* __restrict__ c,
                                       int m,
                                       int n,
                                       int k) {
    constexpr int warps_n = BlockN / kWmmaN;
    constexpr int warps_per_block = kWarpsM * warps_n;
    __shared__ __align__(32) __half a_tile[kBlockTileM * kWmmaK];
    __shared__ __align__(32) __half b_tile[kWmmaK * BlockN];
    __shared__ __align__(32) float
        c_tiles[warps_per_block * kWmmaM * kWmmaN];

    const int thread_id = static_cast<int>(threadIdx.x);
    const int warp_id = thread_id / kWarpSize;
    const int lane = thread_id % kWarpSize;
    const int warp_row = warp_id / warps_n;
    const int warp_col = warp_id % warps_n;
    const int block_row = static_cast<int>(blockIdx.y) * kBlockTileM;
    const int block_col = static_cast<int>(blockIdx.x) * BlockN;

    wmma::fragment<wmma::matrix_a,
                   kWmmaM,
                   kWmmaN,
                   kWmmaK,
                   __half,
                   wmma::row_major>
        a_fragment;
    wmma::fragment<wmma::matrix_b,
                   kWmmaM,
                   kWmmaN,
                   kWmmaK,
                   __half,
                   wmma::row_major>
        b_fragment;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0F);

    const int tile_count = ceil_div(k, kWmmaK);
    for (int tile = 0; tile < tile_count; ++tile) {
        const int k_base = tile * kWmmaK;

        for (int index = thread_id; index < kBlockTileM * kWmmaK;
             index += warps_per_block * kWarpSize) {
            const int local_row = index / kWmmaK;
            const int local_col = index % kWmmaK;
            const int global_row = block_row + local_row;
            const int global_col = k_base + local_col;
            a_tile[index] =
                (global_row < m && global_col < k)
                    ? a[static_cast<std::size_t>(global_row) * k + global_col]
                    : __float2half_rn(0.0F);
        }

        for (int index = thread_id; index < kWmmaK * BlockN;
             index += warps_per_block * kWarpSize) {
            const int local_row = index / BlockN;
            const int local_col = index % BlockN;
            const int global_row = k_base + local_row;
            const int global_col = block_col + local_col;
            b_tile[index] =
                (global_row < k && global_col < n)
                    ? b[static_cast<std::size_t>(global_row) * n + global_col]
                    : __float2half_rn(0.0F);
        }

        __syncthreads();
        wmma::load_matrix_sync(
            a_fragment, &a_tile[warp_row * kWmmaM * kWmmaK], kWmmaK);
        wmma::load_matrix_sync(
            b_fragment, &b_tile[warp_col * kWmmaN], BlockN);
        wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
        __syncthreads();
    }

    float* warp_c = &c_tiles[warp_id * kWmmaM * kWmmaN];
    wmma::store_matrix_sync(warp_c, accumulator, kWmmaN, wmma::mem_row_major);
    __syncthreads();

    const int output_row = block_row + warp_row * kWmmaM;
    const int output_col = block_col + warp_col * kWmmaN;
    for (int index = lane; index < kWmmaM * kWmmaN; index += kWarpSize) {
        const int local_row = index / kWmmaN;
        const int local_col = index % kWmmaN;
        const int global_row = output_row + local_row;
        const int global_col = output_col + local_col;
        if (global_row < m && global_col < n) {
            c[static_cast<std::size_t>(global_row) * n + global_col] =
                __float2half_rn(warp_c[index]);
        }
    }
}

template <int CopyBytes>
__global__ void gemm_wmma_async_kernel(const __half* __restrict__ a,
                                       const __half* __restrict__ b,
                                       __half* __restrict__ c,
                                       int m,
                                       int n,
                                       int k) {
    __shared__ __align__(32)
        __half a_tiles[2][kBlockTileM * kWmmaK];
    __shared__ __align__(32)
        __half b_tiles[2][kWmmaK * kBlockTileN];
    __shared__ __align__(32) float
        c_tiles[kWarpsPerBlock * kWmmaM * kWmmaN];

    const int thread_id = static_cast<int>(threadIdx.x);
    const int warp_id = thread_id / kWarpSize;
    const int lane = thread_id % kWarpSize;
    const int warp_row = warp_id / kWarpsN;
    const int warp_col = warp_id % kWarpsN;
    const int block_row = static_cast<int>(blockIdx.y) * kBlockTileM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockTileN;

    wmma::fragment<wmma::matrix_a,
                   kWmmaM,
                   kWmmaN,
                   kWmmaK,
                   __half,
                   wmma::row_major>
        a_fragment;
    wmma::fragment<wmma::matrix_b,
                   kWmmaM,
                   kWmmaN,
                   kWmmaK,
                   __half,
                   wmma::row_major>
        b_fragment;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
        accumulator;
    wmma::fill_fragment(accumulator, 0.0F);

    const auto issue_tile_copy = [&](int tile, int stage) {
        const int k_base = tile * kWmmaK;
        constexpr int elements_per_copy = CopyBytes / sizeof(__half);
        constexpr int a_copies_per_row = kWmmaK / elements_per_copy;
        constexpr int a_copy_count = kBlockTileM * a_copies_per_row;
        for (int copy = thread_id; copy < a_copy_count;
             copy += kWarpsPerBlock * kWarpSize) {
            const int local_row = copy / a_copies_per_row;
            const int local_col =
                (copy % a_copies_per_row) * elements_per_copy;
            const int global_row = block_row + local_row;
            const int global_col = k_base + local_col;
            int valid_bytes = 0;
            const __half* source = a;
            if (global_row < m && global_col < k) {
                const int remaining = k - global_col;
                valid_bytes = remaining >= elements_per_copy
                                  ? CopyBytes
                                  : remaining * static_cast<int>(sizeof(__half));
                source = &a[static_cast<std::size_t>(global_row) * k + global_col];
            }
            cp_async_copy<CopyBytes>(
                &a_tiles[stage][local_row * kWmmaK + local_col],
                source,
                valid_bytes);
        }

        constexpr int b_copies_per_row = kBlockTileN / elements_per_copy;
        constexpr int b_copy_count = kWmmaK * b_copies_per_row;
        for (int copy = thread_id; copy < b_copy_count;
             copy += kWarpsPerBlock * kWarpSize) {
            const int local_row = copy / b_copies_per_row;
            const int local_col =
                (copy % b_copies_per_row) * elements_per_copy;
            const int global_row = k_base + local_row;
            const int global_col = block_col + local_col;
            int valid_bytes = 0;
            const __half* source = b;
            if (global_row < k && global_col < n) {
                const int remaining = n - global_col;
                valid_bytes = remaining >= elements_per_copy
                                  ? CopyBytes
                                  : remaining * static_cast<int>(sizeof(__half));
                source = &b[static_cast<std::size_t>(global_row) * n + global_col];
            }
            cp_async_copy<CopyBytes>(
                &b_tiles[stage][local_row * kBlockTileN + local_col],
                source,
                valid_bytes);
        }
    };

    const int tile_count = ceil_div(k, kWmmaK);
    issue_tile_copy(0, 0);
    cp_async_commit();

    for (int tile = 0; tile < tile_count; ++tile) {
        const int stage = tile & 1;
        cp_async_wait_all();
        __syncthreads();

        if (tile + 1 < tile_count) {
            issue_tile_copy(tile + 1, stage ^ 1);
            cp_async_commit();
        }

        wmma::load_matrix_sync(
            a_fragment,
            &a_tiles[stage][warp_row * kWmmaM * kWmmaK],
            kWmmaK);
        wmma::load_matrix_sync(
            b_fragment,
            &b_tiles[stage][warp_col * kWmmaN],
            kBlockTileN);
        wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }

    float* warp_c = &c_tiles[warp_id * kWmmaM * kWmmaN];
    wmma::store_matrix_sync(warp_c, accumulator, kWmmaN, wmma::mem_row_major);
    __syncthreads();

    const int output_row = block_row + warp_row * kWmmaM;
    const int output_col = block_col + warp_col * kWmmaN;
    for (int index = lane; index < kWmmaM * kWmmaN; index += kWarpSize) {
        const int local_row = index / kWmmaN;
        const int local_col = index % kWmmaN;
        const int global_row = output_row + local_row;
        const int global_col = output_col + local_col;
        if (global_row < m && global_col < n) {
            c[static_cast<std::size_t>(global_row) * n + global_col] =
                __float2half_rn(warp_c[index]);
        }
    }
}

}  // namespace

void launch_gemm_wmma(const __half* a,
                      const __half* b,
                      __half* c,
                      int m,
                      int n,
                      int k,
                      cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    const dim3 block(kWarpSize);
    const dim3 grid(ceil_div(n, kWmmaN), ceil_div(m, kWmmaM));
    gemm_wmma_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_KERNEL_CHECK();
}

void launch_gemm_wmma_block(const __half* a,
                            const __half* b,
                            __half* c,
                            int m,
                            int n,
                            int k,
                            cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    const dim3 block(kWarpsPerBlock * kWarpSize);
    const dim3 grid(ceil_div(n, kBlockTileN), ceil_div(m, kBlockTileM));
    gemm_wmma_block_kernel<kBlockTileN>
        <<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_KERNEL_CHECK();
}

void launch_gemm_wmma_block_64(const __half* a,
                               const __half* b,
                               __half* c,
                               int m,
                               int n,
                               int k,
                               cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    constexpr int block_n = 64;
    constexpr int warp_count = kWarpsM * (block_n / kWmmaN);
    const dim3 block(warp_count * kWarpSize);
    const dim3 grid(ceil_div(n, block_n), ceil_div(m, kBlockTileM));
    gemm_wmma_block_kernel<block_n>
        <<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_KERNEL_CHECK();
}

void launch_gemm_wmma_async(const __half* a,
                            const __half* b,
                            __half* c,
                            int m,
                            int n,
                            int k,
                            cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    // cp.async copies 4 bytes per instruction. Even leading dimensions keep
    // every row aligned; odd strides use the fully general synchronous path.
    if ((k & 1) != 0 || (n & 1) != 0) {
        launch_gemm_wmma_block(a, b, c, m, n, k, stream);
        return;
    }

    const dim3 block(kWarpsPerBlock * kWarpSize);
    const dim3 grid(ceil_div(n, kBlockTileN), ceil_div(m, kBlockTileM));
    if ((k & 7) == 0 && (n & 7) == 0) {
        gemm_wmma_async_kernel<16>
            <<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    } else {
        gemm_wmma_async_kernel<4>
            <<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    }
    CUDA_KERNEL_CHECK();
}

}  // namespace cuda_gemm
