#include "gemm.h"

namespace cuda_gemm {

GemmKernel select_gemm_kernel(int m, int n, int k) {
    constexpr int kSmallMUpperBound = 32;
    const bool ptx_fast_path = (n & 7) == 0 && (k & 7) == 0;
    if (m > 0 && m <= kSmallMUpperBound && ptx_fast_path) {
        return GemmKernel::kMmaPtx;
    }
    return GemmKernel::kWmmaAsync;
}

void launch_gemm_dispatch(const __half* a,
                          const __half* b,
                          __half* c,
                          int m,
                          int n,
                          int k,
                          cudaStream_t stream) {
    switch (select_gemm_kernel(m, n, k)) {
        case GemmKernel::kMmaPtx:
            launch_gemm_mma_ptx(a, b, c, m, n, k, stream);
            break;
        case GemmKernel::kWmmaAsync:
            launch_gemm_wmma_async(a, b, c, m, n, k, stream);
            break;
    }
}

}  // namespace cuda_gemm
