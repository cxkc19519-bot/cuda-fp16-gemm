#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace cuda_gemm {

struct GemmShape {
    int m;
    int n;
    int k;
};

struct ErrorMetrics {
    float max_absolute_error = 0.0F;
    float max_relative_error = 0.0F;
    std::size_t max_absolute_error_index = 0;
    std::size_t max_relative_error_index = 0;
    std::size_t mismatch_count = 0;
};

std::vector<__half> make_random_fp16(std::size_t count,
                                     std::uint64_t seed,
                                     float minimum = -0.5F,
                                     float maximum = 0.5F);

std::vector<__half> cpu_reference_gemm(const std::vector<__half>& a,
                                       const std::vector<__half>& b,
                                       const GemmShape& shape);

ErrorMetrics compare_fp16(const std::vector<__half>& actual,
                          const std::vector<__half>& expected,
                          float absolute_tolerance,
                          float relative_tolerance);

float measure_cuda_time_us(const std::function<void()>& launch,
                           cudaStream_t stream,
                           int warmup_iterations,
                           int measured_iterations);

double gemm_tflops(const GemmShape& shape, float latency_us);

}  // namespace cuda_gemm
