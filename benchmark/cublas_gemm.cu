#include "benchmark.h"
#include "common.cuh"
#include "gemm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace cuda_gemm {

void launch_cublas_gemm(cublasHandle_t handle,
                        const __half* a,
                        const __half* b,
                        __half* c,
                        int m,
                        int n,
                        int k) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;

    // Row-major C = A * B is equivalent in memory to the column-major
    // operation C^T = B^T * A^T. Row-major B[K, N] is viewed as a
    // column-major B^T[N, K], and row-major A[M, K] as A^T[K, M].
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              m,
                              k,
                              &alpha,
                              b,
                              CUDA_R_16F,
                              n,
                              a,
                              CUDA_R_16F,
                              k,
                              &beta,
                              c,
                              CUDA_R_16F,
                              n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

std::vector<__half> make_random_fp16(std::size_t count,
                                     std::uint64_t seed,
                                     float minimum,
                                     float maximum) {
    if (minimum > maximum) {
        throw std::invalid_argument("minimum must not exceed maximum");
    }

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> distribution(minimum, maximum);
    std::vector<__half> values(count);
    std::generate(values.begin(), values.end(), [&]() {
        return __float2half_rn(distribution(generator));
    });
    return values;
}

std::vector<__half> cpu_reference_gemm(const std::vector<__half>& a,
                                       const std::vector<__half>& b,
                                       const GemmShape& shape) {
    const std::size_t expected_a_size = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t expected_b_size = static_cast<std::size_t>(shape.k) * shape.n;
    if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0 ||
        a.size() != expected_a_size || b.size() != expected_b_size) {
        throw std::invalid_argument("invalid CPU reference GEMM inputs");
    }

    std::vector<__half> c(static_cast<std::size_t>(shape.m) * shape.n);
    for (int row = 0; row < shape.m; ++row) {
        for (int col = 0; col < shape.n; ++col) {
            float accumulator = 0.0F;
            for (int inner = 0; inner < shape.k; ++inner) {
                const float a_value = __half2float(a[static_cast<std::size_t>(row) * shape.k + inner]);
                const float b_value = __half2float(b[static_cast<std::size_t>(inner) * shape.n + col]);
                accumulator = std::fma(a_value, b_value, accumulator);
            }
            c[static_cast<std::size_t>(row) * shape.n + col] = __float2half_rn(accumulator);
        }
    }
    return c;
}

ErrorMetrics compare_fp16(const std::vector<__half>& actual,
                          const std::vector<__half>& expected,
                          float absolute_tolerance,
                          float relative_tolerance) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("actual and expected sizes differ");
    }
    if (absolute_tolerance < 0.0F || relative_tolerance < 0.0F) {
        throw std::invalid_argument("tolerances must be non-negative");
    }

    ErrorMetrics metrics;
    constexpr float relative_floor = 1.0e-6F;

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float actual_value = __half2float(actual[index]);
        const float expected_value = __half2float(expected[index]);
        const float absolute_error = std::abs(actual_value - expected_value);
        const float relative_error = absolute_error /
                                     std::max(std::abs(expected_value), relative_floor);

        if (absolute_error > metrics.max_absolute_error) {
            metrics.max_absolute_error = absolute_error;
            metrics.max_absolute_error_index = index;
        }
        if (relative_error > metrics.max_relative_error) {
            metrics.max_relative_error = relative_error;
            metrics.max_relative_error_index = index;
        }

        const float allowed_error = absolute_tolerance +
                                    relative_tolerance * std::abs(expected_value);
        if (!std::isfinite(actual_value) || absolute_error > allowed_error) {
            ++metrics.mismatch_count;
        }
    }

    return metrics;
}

float measure_cuda_time_us(const std::function<void()>& launch,
                           cudaStream_t stream,
                           int warmup_iterations,
                           int measured_iterations) {
    if (warmup_iterations < 0 || measured_iterations <= 0) {
        throw std::invalid_argument("invalid benchmark iteration count");
    }

    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        launch();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iteration = 0; iteration < measured_iterations; ++iteration) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_milliseconds * 1000.0F / static_cast<float>(measured_iterations);
}

double gemm_tflops(const GemmShape& shape, float latency_us) {
    if (latency_us <= 0.0F) {
        return 0.0;
    }

    const double operations = 2.0 * static_cast<double>(shape.m) *
                              static_cast<double>(shape.n) *
                              static_cast<double>(shape.k);
    return operations / (static_cast<double>(latency_us) * 1.0e6);
}

}  // namespace cuda_gemm
