#pragma once

#include "benchmark.h"

#include <array>

namespace cuda_gemm {

inline constexpr std::array<GemmShape, 4> kPhaseOneBenchmarkShapes{{
    {128, 128, 128},
    {512, 512, 512},
    {1024, 1024, 1024},
    {257, 511, 1025},
}};

// LLM-oriented shapes from the project task book. N and K model a common
// hidden dimension, while M separates token-by-token decode from prefill.
inline constexpr std::array<GemmShape, 8> kLlmDecodeShapes{{
    {1, 4096, 4096},
    {2, 4096, 4096},
    {4, 4096, 4096},
    {8, 4096, 4096},
    {16, 4096, 4096},
    {32, 4096, 4096},
    {64, 4096, 4096},
    {128, 4096, 4096},
}};

inline constexpr std::array<GemmShape, 6> kLlmPrefillShapes{{
    {128, 4096, 4096},
    {512, 4096, 4096},
    {1024, 4096, 4096},
    {2048, 4096, 4096},
    {4096, 4096, 4096},
    {8192, 4096, 4096},
}};

}  // namespace cuda_gemm
