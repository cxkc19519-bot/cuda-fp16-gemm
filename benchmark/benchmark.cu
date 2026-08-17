#include "benchmark.h"
#include "common.cuh"
#include "gemm.h"
#include "shapes.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
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

struct Options {
    int warmup_iterations = 20;
    int measured_iterations = 100;
    std::string csv_path;
    std::string kernel = "all";
    std::string suite = "default";
    std::optional<cuda_gemm::GemmShape> shape;
};

struct Result {
    std::string kernel;
    cuda_gemm::GemmShape shape;
    float latency_us;
    double tflops;
    double cublas_ratio;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--warmup" && index + 1 < argc) {
            options.warmup_iterations = std::stoi(argv[++index]);
        } else if (argument == "--iterations" && index + 1 < argc) {
            options.measured_iterations = std::stoi(argv[++index]);
        } else if (argument == "--csv" && index + 1 < argc) {
            options.csv_path = argv[++index];
        } else if (argument == "--kernel" && index + 1 < argc) {
            options.kernel = argv[++index];
        } else if (argument == "--suite" && index + 1 < argc) {
            options.suite = argv[++index];
        } else if (argument == "--shape" && index + 3 < argc) {
            options.shape = cuda_gemm::GemmShape{
                std::stoi(argv[++index]), std::stoi(argv[++index]), std::stoi(argv[++index])};
        } else if (argument == "--help") {
            std::cout << "Usage: gemm_benchmark [--warmup N] [--iterations N] [--csv PATH]"
                         " [--kernel NAME] [--suite NAME] [--shape M N K]\n"
                         "Suites: default, llm-decode, llm-prefill, llm-all\n"
                         "Kernels: all, dispatch-candidates, final-comparison, naive, coalesced, shared32, shared16, register,"
                         " reg_f32sm, reg_inter, reg_vec, reg_vec4, wmma, wmma_block,"
                         " wmma_64, wmma_async, mma_ptx, dispatch, cublas\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown or incomplete option: " + argument);
        }
    }

    if (options.warmup_iterations < 0 || options.measured_iterations <= 0) {
        throw std::invalid_argument("warmup must be non-negative and iterations must be positive");
    }
    const std::vector<std::string> kernels{
        "all", "dispatch-candidates", "final-comparison", "naive", "coalesced", "shared32", "shared16", "register", "reg_f32sm",
        "reg_inter", "reg_vec", "reg_vec4", "wmma", "wmma_block", "wmma_64",
        "wmma_async", "mma_ptx", "dispatch", "cublas"};
    if (std::find(kernels.begin(), kernels.end(), options.kernel) == kernels.end()) {
        throw std::invalid_argument("unknown kernel: " + options.kernel);
    }
    const std::vector<std::string> suites{
        "default", "llm-decode", "llm-prefill", "llm-all"};
    if (std::find(suites.begin(), suites.end(), options.suite) == suites.end()) {
        throw std::invalid_argument("unknown suite: " + options.suite);
    }
    if (options.shape.has_value() && options.suite != "default") {
        throw std::invalid_argument("--shape and --suite cannot be used together");
    }
    if (options.shape.has_value() &&
        (options.shape->m <= 0 || options.shape->n <= 0 || options.shape->k <= 0)) {
        throw std::invalid_argument("shape dimensions must be positive");
    }
    return options;
}

std::vector<Result> benchmark_shape(const cuda_gemm::GemmShape& shape,
                                    cublasHandle_t handle,
                                    cudaStream_t stream,
                                    const Options& options) {
    const std::size_t a_count = static_cast<std::size_t>(shape.m) * shape.k;
    const std::size_t b_count = static_cast<std::size_t>(shape.k) * shape.n;
    const std::size_t c_count = static_cast<std::size_t>(shape.m) * shape.n;

    const auto host_a = cuda_gemm::make_random_fp16(a_count, 1009);
    const auto host_b = cuda_gemm::make_random_fp16(b_count, 2027);

    DeviceBuffer<__half> device_a(a_count);
    DeviceBuffer<__half> device_b(b_count);
    DeviceBuffer<__half> device_c(c_count);

    CUDA_CHECK(cudaMemcpyAsync(device_a.get(), host_a.data(), device_a.bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_b.get(), host_b.data(), device_b.bytes(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const float cublas_latency = cuda_gemm::measure_cuda_time_us(
        [&]() {
            cuda_gemm::launch_cublas_gemm(handle, device_a.get(), device_b.get(), device_c.get(),
                                          shape.m, shape.n, shape.k);
        },
        stream,
        options.warmup_iterations,
        options.measured_iterations);

    const double cublas_tflops = cuda_gemm::gemm_tflops(shape, cublas_latency);
    const auto measure_kernel = [&](const std::string& name, const std::function<void()>& launch) {
        const float latency = cuda_gemm::measure_cuda_time_us(
            launch, stream, options.warmup_iterations, options.measured_iterations);
        const double tflops = cuda_gemm::gemm_tflops(shape, latency);
        return Result{name,
                      shape,
                      latency,
                      tflops,
                      cublas_tflops > 0.0 ? tflops / cublas_tflops : 0.0};
    };

    const auto selected = [&](const std::string& name) {
        const bool dispatch_candidate =
            name == "coalesced" || name == "reg_vec" || name == "wmma_block" ||
            name == "wmma_async" || name == "mma_ptx" || name == "cublas";
        const bool final_comparison =
            name == "reg_vec" || name == "wmma_block" || name == "wmma_async" ||
            name == "mma_ptx" || name == "dispatch" || name == "cublas";
        return options.kernel == "all" || options.kernel == name ||
               (options.kernel == "dispatch-candidates" && dispatch_candidate) ||
               (options.kernel == "final-comparison" && final_comparison);
    };

    std::vector<Result> results;

    if (selected("naive")) {
        results.push_back(measure_kernel("naive", [&]() {
            cuda_gemm::launch_gemm_naive(device_a.get(), device_b.get(), device_c.get(),
                                         shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("coalesced")) {
        results.push_back(measure_kernel("coalesced", [&]() {
            cuda_gemm::launch_gemm_coalesced(device_a.get(), device_b.get(), device_c.get(),
                                             shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("shared32")) {
        results.push_back(measure_kernel("shared32", [&]() {
            cuda_gemm::launch_gemm_shared(device_a.get(), device_b.get(), device_c.get(),
                                          shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("shared16")) {
        results.push_back(measure_kernel("shared16", [&]() {
            cuda_gemm::launch_gemm_shared_16(device_a.get(), device_b.get(), device_c.get(),
                                             shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("register")) {
        results.push_back(measure_kernel("register", [&]() {
            cuda_gemm::launch_gemm_register(device_a.get(), device_b.get(), device_c.get(),
                                            shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("reg_f32sm")) {
        results.push_back(measure_kernel("reg_f32sm", [&]() {
            cuda_gemm::launch_gemm_register_fp32_smem(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("reg_inter")) {
        results.push_back(measure_kernel("reg_inter", [&]() {
            cuda_gemm::launch_gemm_register_interleaved(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("reg_vec")) {
        results.push_back(measure_kernel("reg_vec", [&]() {
            cuda_gemm::launch_gemm_register_vectorized(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("reg_vec4")) {
        results.push_back(measure_kernel("reg_vec4", [&]() {
            cuda_gemm::launch_gemm_register_vectorized_store(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("wmma")) {
        results.push_back(measure_kernel("wmma", [&]() {
            cuda_gemm::launch_gemm_wmma(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("wmma_block")) {
        results.push_back(measure_kernel("wmma_block", [&]() {
            cuda_gemm::launch_gemm_wmma_block(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("wmma_64")) {
        results.push_back(measure_kernel("wmma_64", [&]() {
            cuda_gemm::launch_gemm_wmma_block_64(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("wmma_async")) {
        results.push_back(measure_kernel("wmma_async", [&]() {
            cuda_gemm::launch_gemm_wmma_async(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("mma_ptx")) {
        results.push_back(measure_kernel("mma_ptx", [&]() {
            cuda_gemm::launch_gemm_mma_ptx(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("dispatch")) {
        results.push_back(measure_kernel("dispatch", [&]() {
            cuda_gemm::launch_gemm_dispatch(
                device_a.get(), device_b.get(), device_c.get(),
                shape.m, shape.n, shape.k, stream);
        }));
    }
    if (selected("cublas")) {
        results.push_back({"cublas", shape, cublas_latency, cublas_tflops, 1.0});
    }
    return results;
}

void print_results(const std::vector<Result>& results) {
    std::cout << std::left << std::setw(10) << "Kernel"
              << std::right << std::setw(7) << "M"
              << std::setw(7) << "N"
              << std::setw(7) << "K"
              << std::setw(16) << "Latency(us)"
              << std::setw(13) << "TFLOPS"
              << std::setw(14) << "vs cuBLAS" << '\n';
    std::cout << std::string(74, '-') << '\n';

    for (const auto& result : results) {
        std::cout << std::left << std::setw(10) << result.kernel
                  << std::right << std::setw(7) << result.shape.m
                  << std::setw(7) << result.shape.n
                  << std::setw(7) << result.shape.k
                  << std::setw(16) << std::fixed << std::setprecision(3) << result.latency_us
                  << std::setw(13) << std::setprecision(3) << result.tflops
                  << std::setw(13) << std::setprecision(1) << result.cublas_ratio * 100.0 << "%\n";
    }
}

void write_csv(const std::string& path, const std::vector<Result>& results) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("could not open CSV output: " + path);
    }

    output << "kernel,M,N,K,latency_us,tflops,cublas_ratio\n";
    output << std::setprecision(9);
    for (const auto& result : results) {
        output << result.kernel << ',' << result.shape.m << ',' << result.shape.n << ','
               << result.shape.k << ',' << result.latency_us << ',' << result.tflops << ','
               << result.cublas_ratio << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::cout << "Device: " << properties.name << " (sm_" << properties.major
                  << properties.minor << ")\n";
        std::cout << "Warmup: " << options.warmup_iterations
                  << ", measured iterations: " << options.measured_iterations << "\n\n";

        cudaStream_t stream = nullptr;
        cublasHandle_t handle = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));

        std::vector<Result> results;
        std::vector<cuda_gemm::GemmShape> shapes;
        if (options.shape.has_value()) {
            shapes.push_back(*options.shape);
        } else if (options.suite == "llm-decode") {
            shapes.assign(cuda_gemm::kLlmDecodeShapes.begin(), cuda_gemm::kLlmDecodeShapes.end());
        } else if (options.suite == "llm-prefill") {
            shapes.assign(cuda_gemm::kLlmPrefillShapes.begin(), cuda_gemm::kLlmPrefillShapes.end());
        } else if (options.suite == "llm-all") {
            shapes.assign(cuda_gemm::kLlmDecodeShapes.begin(), cuda_gemm::kLlmDecodeShapes.end());
            shapes.insert(shapes.end(), cuda_gemm::kLlmPrefillShapes.begin() + 1,
                          cuda_gemm::kLlmPrefillShapes.end());
        } else {
            shapes.assign(cuda_gemm::kPhaseOneBenchmarkShapes.begin(),
                          cuda_gemm::kPhaseOneBenchmarkShapes.end());
        }
        for (const auto& shape : shapes) {
            auto shape_results = benchmark_shape(shape, handle, stream, options);
            results.insert(results.end(), shape_results.begin(), shape_results.end());
        }

        print_results(results);
        if (!options.csv_path.empty()) {
            write_csv(options.csv_path, results);
            std::cout << "\nCSV written to: " << options.csv_path << '\n';
        }

        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
