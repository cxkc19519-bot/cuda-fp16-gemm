#include "benchmark.h"
#include "common.cuh"
#include "gemm.h"

#include <cuda_fp16.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

namespace {

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return data_; }
    const T* get() const { return data_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

using GemmLauncher = void (*)(const __half*,
                              const __half*,
                              __half*,
                              int,
                              int,
                              int,
                              cudaStream_t);

bool run_shape(const cuda_gemm::GemmShape& shape, cublasHandle_t handle, cudaStream_t stream) {
    const std::size_t a_count = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t b_count = static_cast<std::size_t>(shape.k) * shape.n;
    const std::size_t c_count = static_cast<std::size_t>(shape.m) * shape.n;

    const auto host_a = cuda_gemm::make_random_fp16(a_count, 17);
    const auto host_b = cuda_gemm::make_random_fp16(b_count, 29);
    std::vector<__half> host_actual(c_count);
    std::vector<__half> host_expected(c_count);

    DeviceBuffer<__half> device_a(a_count);
    DeviceBuffer<__half> device_b(b_count);
    DeviceBuffer<__half> device_actual(c_count);
    DeviceBuffer<__half> device_expected(c_count);

    CUDA_CHECK(cudaMemcpyAsync(device_a.get(), host_a.data(), device_a.bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_b.get(), host_b.data(), device_b.bytes(),
                               cudaMemcpyHostToDevice, stream));

    cuda_gemm::launch_cublas_gemm(handle, device_a.get(), device_b.get(),
                                  device_expected.get(), shape.m, shape.n, shape.k);
    CUDA_CHECK(cudaMemcpyAsync(host_expected.data(), device_expected.get(), device_expected.bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    constexpr float absolute_tolerance = 2.0e-2F;
    constexpr float relative_tolerance = 2.0e-2F;
    const std::vector<std::pair<const char*, GemmLauncher>> kernels{
        {"naive", cuda_gemm::launch_gemm_naive},
        {"coalesced", cuda_gemm::launch_gemm_coalesced},
        {"shared32", cuda_gemm::launch_gemm_shared},
        {"shared16", cuda_gemm::launch_gemm_shared_16},
        {"register", cuda_gemm::launch_gemm_register},
        {"reg_f32sm", cuda_gemm::launch_gemm_register_fp32_smem},
        {"reg_inter", cuda_gemm::launch_gemm_register_interleaved},
        {"reg_vec", cuda_gemm::launch_gemm_register_vectorized},
        {"reg_vec4", cuda_gemm::launch_gemm_register_vectorized_store},
        {"wmma", cuda_gemm::launch_gemm_wmma},
        {"wmma_block", cuda_gemm::launch_gemm_wmma_block},
        {"wmma_64", cuda_gemm::launch_gemm_wmma_block_64},
        {"wmma_async", cuda_gemm::launch_gemm_wmma_async},
        {"mma_ptx", cuda_gemm::launch_gemm_mma_ptx},
        {"dispatch", cuda_gemm::launch_gemm_dispatch},
    };

    bool shape_passed = true;
    for (const auto& [name, launch] : kernels) {
        launch(device_a.get(), device_b.get(), device_actual.get(),
               shape.m, shape.n, shape.k, stream);
        CUDA_CHECK(cudaMemcpyAsync(host_actual.data(), device_actual.get(), device_actual.bytes(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const auto metrics = cuda_gemm::compare_fp16(host_actual, host_expected,
                                                     absolute_tolerance, relative_tolerance);
        const bool passed = metrics.mismatch_count == 0;
        shape_passed = passed && shape_passed;

        std::cout << (passed ? "PASS" : "FAIL") << "  " << std::left << std::setw(10) << name
                  << std::right << " M=" << std::setw(4) << shape.m
                  << " N=" << std::setw(4) << shape.n << " K=" << std::setw(4) << shape.k
                  << "  max_abs=" << metrics.max_absolute_error
                  << "  max_rel=" << metrics.max_relative_error
                  << "  mismatches=" << metrics.mismatch_count << '\n';
    }
    return shape_passed;
}

bool verify_cublas_row_major_contract(cublasHandle_t handle, cudaStream_t stream) {
    constexpr cuda_gemm::GemmShape shape{3, 5, 7};
    const auto host_a = cuda_gemm::make_random_fp16(
        static_cast<std::size_t>(shape.m) * shape.k, 101);
    const auto host_b = cuda_gemm::make_random_fp16(
        static_cast<std::size_t>(shape.k) * shape.n, 103);
    const auto host_cpu = cuda_gemm::cpu_reference_gemm(host_a, host_b, shape);
    std::vector<__half> host_cublas(static_cast<std::size_t>(shape.m) * shape.n);

    DeviceBuffer<__half> device_a(host_a.size());
    DeviceBuffer<__half> device_b(host_b.size());
    DeviceBuffer<__half> device_c(host_cublas.size());
    CUDA_CHECK(cudaMemcpyAsync(device_a.get(), host_a.data(), device_a.bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_b.get(), host_b.data(), device_b.bytes(),
                               cudaMemcpyHostToDevice, stream));
    cuda_gemm::launch_cublas_gemm(handle, device_a.get(), device_b.get(), device_c.get(),
                                  shape.m, shape.n, shape.k);
    CUDA_CHECK(cudaMemcpyAsync(host_cublas.data(), device_c.get(), device_c.bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto metrics = cuda_gemm::compare_fp16(host_cublas, host_cpu, 2.0e-2F, 2.0e-2F);
    const bool passed = metrics.mismatch_count == 0;
    std::cout << (passed ? "PASS" : "FAIL")
              << "  cuBLAS row-major contract (CPU reference, 3x5x7)"
              << "  max_abs=" << metrics.max_absolute_error
              << "  mismatches=" << metrics.mismatch_count << '\n';
    return passed;
}

bool verify_dispatch_policy() {
    const bool passed =
        cuda_gemm::select_gemm_kernel(1, 4096, 4096) == cuda_gemm::GemmKernel::kMmaPtx &&
        cuda_gemm::select_gemm_kernel(32, 4096, 4096) == cuda_gemm::GemmKernel::kMmaPtx &&
        cuda_gemm::select_gemm_kernel(33, 4096, 4096) == cuda_gemm::GemmKernel::kWmmaAsync &&
        cuda_gemm::select_gemm_kernel(16, 4095, 4096) == cuda_gemm::GemmKernel::kWmmaAsync &&
        cuda_gemm::select_gemm_kernel(16, 4096, 4095) == cuda_gemm::GemmKernel::kWmmaAsync;
    std::cout << (passed ? "PASS" : "FAIL")
              << "  shape-aware dispatch policy boundaries\n";
    return passed;
}

}  // namespace

int main() {
    const std::vector<cuda_gemm::GemmShape> shapes{
        {17, 19, 23},
        {16, 64, 64},
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {128, 256, 512},
        {65, 72, 40},
        {65, 70, 34},
        {257, 511, 1025},
    };

    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    bool all_passed = verify_cublas_row_major_contract(handle, stream);
    all_passed = verify_dispatch_policy() && all_passed;
    for (const auto& shape : shapes) {
        all_passed = run_shape(shape, handle, stream) && all_passed;
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (!all_passed) {
        std::cerr << "One or more correctness cases failed.\n";
        return 1;
    }

    std::cout << "All correctness cases passed.\n";
    return 0;
}
