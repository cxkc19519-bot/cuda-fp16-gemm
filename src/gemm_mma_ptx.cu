#include "common.cuh"
#include "gemm.h"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace cuda_gemm {
namespace {

constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;
constexpr int kBlockM = 32;
constexpr int kBlockN = 32;
constexpr int kWarpsM = kBlockM / kMmaM;
constexpr int kWarpsN = kBlockN / kMmaN;
constexpr int kWarpCount = kWarpsM * kWarpsN;
constexpr int kWarpSize = 32;
constexpr int kThreadCount = kWarpCount * kWarpSize;

__device__ __forceinline__ unsigned int shared_address(const void* pointer) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void cp_async_16(void* shared_destination,
                                           const void* global_source,
                                           int valid_bytes) {
    const unsigned int destination = shared_address(shared_destination);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                     "r"(destination),
                     "l"(global_source),
                     "r"(valid_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void ldmatrix_x4(std::uint32_t (&registers)[4],
                                            const __half* pointer) {
    const unsigned int address = shared_address(pointer);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(registers[0]), "=r"(registers[1]),
          "=r"(registers[2]), "=r"(registers[3])
        : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    std::uint32_t (&registers)[2], const __half* pointer) {
    const unsigned int address = shared_address(pointer);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(registers[0]), "=r"(registers[1])
        : "r"(address));
}

__device__ __forceinline__ void mma_m16n8k16(
    float (&accumulators)[4],
    const std::uint32_t (&a_registers)[4],
    const std::uint32_t (&b_registers)[2]) {
    float d0;
    float d1;
    float d2;
    float d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a_registers[0]), "r"(a_registers[1]),
          "r"(a_registers[2]), "r"(a_registers[3]),
          "r"(b_registers[0]), "r"(b_registers[1]),
          "f"(accumulators[0]), "f"(accumulators[1]),
          "f"(accumulators[2]), "f"(accumulators[3]));
    accumulators[0] = d0;
    accumulators[1] = d1;
    accumulators[2] = d2;
    accumulators[3] = d3;
}

__global__ void gemm_mma_ptx_kernel(const __half* __restrict__ a,
                                    const __half* __restrict__ b,
                                    __half* __restrict__ c,
                                    int m,
                                    int n,
                                    int k) {
    __shared__ __align__(32) __half a_tiles[2][kBlockM * kMmaK];
    __shared__ __align__(32) __half b_tiles[2][kMmaK * kBlockN];

    const int thread_id = static_cast<int>(threadIdx.x);
    const int warp_id = thread_id / kWarpSize;
    const int lane = thread_id % kWarpSize;
    const int warp_row = warp_id / kWarpsN;
    const int warp_col = warp_id % kWarpsN;
    const int block_row = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_col = static_cast<int>(blockIdx.x) * kBlockN;

    float accumulators[4] = {};

    const auto issue_tile_copy = [&](int tile, int stage) {
        constexpr int elements_per_copy = 8;
        constexpr int a_copies_per_row = kMmaK / elements_per_copy;
        constexpr int a_copy_count = kBlockM * a_copies_per_row;
        if (thread_id < a_copy_count) {
            const int local_row = thread_id / a_copies_per_row;
            const int local_col =
                (thread_id % a_copies_per_row) * elements_per_copy;
            const int global_row = block_row + local_row;
            const int global_col = tile * kMmaK + local_col;
            int valid_bytes = 0;
            const __half* source = a;
            if (global_row < m && global_col < k) {
                const int remaining = k - global_col;
                valid_bytes = remaining >= elements_per_copy ? 16 : remaining * 2;
                source = &a[static_cast<std::size_t>(global_row) * k + global_col];
            }
            cp_async_16(&a_tiles[stage][local_row * kMmaK + local_col],
                        source,
                        valid_bytes);
        }

        constexpr int b_copies_per_row = kBlockN / elements_per_copy;
        constexpr int b_copy_count = kMmaK * b_copies_per_row;
        if (thread_id < b_copy_count) {
            const int local_row = thread_id / b_copies_per_row;
            const int local_col =
                (thread_id % b_copies_per_row) * elements_per_copy;
            const int global_row = tile * kMmaK + local_row;
            const int global_col = block_col + local_col;
            int valid_bytes = 0;
            const __half* source = b;
            if (global_row < k && global_col < n) {
                const int remaining = n - global_col;
                valid_bytes = remaining >= elements_per_copy ? 16 : remaining * 2;
                source = &b[static_cast<std::size_t>(global_row) * n + global_col];
            }
            cp_async_16(&b_tiles[stage][local_row * kBlockN + local_col],
                        source,
                        valid_bytes);
        }
    };

    const int tile_count = ceil_div(k, kMmaK);
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

        const int a_matrix = lane / 8;
        const int a_row = (a_matrix & 1) * 8 + (lane & 7);
        const int a_col = (a_matrix / 2) * 8;
        const __half* a_pointer =
            &a_tiles[stage][(warp_row * kMmaM + a_row) * kMmaK + a_col];

        const int b_lane = lane & 15;
        const int b_row = (b_lane / 8) * 8 + (b_lane & 7);
        const __half* b_pointer =
            &b_tiles[stage][b_row * kBlockN + warp_col * kMmaN];

        std::uint32_t a_registers[4];
        std::uint32_t b_registers[2];
        ldmatrix_x4(a_registers, a_pointer);
        ldmatrix_x2_trans(b_registers, b_pointer);
        mma_m16n8k16(accumulators, a_registers, b_registers);
    }

    const int lane_group = lane / 4;
    const int thread_in_group = lane & 3;
    const int output_col = block_col + warp_col * kMmaN + thread_in_group * 2;
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        const int output_row = block_row + warp_row * kMmaM + lane_group +
                               (index >= 2 ? 8 : 0);
        const int column = output_col + (index & 1);
        if (output_row < m && column < n) {
            c[static_cast<std::size_t>(output_row) * n + column] =
                __float2half_rn(accumulators[index]);
        }
    }
}

}  // namespace

void launch_gemm_mma_ptx(const __half* a,
                         const __half* b,
                         __half* c,
                         int m,
                         int n,
                         int k,
                         cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    if ((k & 7) != 0 || (n & 7) != 0) {
        launch_gemm_wmma_async(a, b, c, m, n, k, stream);
        return;
    }

    const dim3 block(kThreadCount);
    const dim3 grid(ceil_div(n, kBlockN), ceil_div(m, kBlockM));
    gemm_mma_ptx_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    CUDA_KERNEL_CHECK();
}

}  // namespace cuda_gemm
