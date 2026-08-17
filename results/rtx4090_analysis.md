# RTX 4090（sm_89）验证与性能分析

## 环境

- GPU: NVIDIA GeForce RTX 4090 24 GB
- Compute Capability: 8.9
- CUDA Toolkit: 12.4
- Driver: 580.173.02
- CMake: 3.22.1
- Build: Release, `CMAKE_CUDA_ARCHITECTURES=89`
- Benchmark: GPU 1，20 warmup + 100 measured iterations

全部正确性测试通过。Benchmark 固定 `N=K=4096`，覆盖任务书中的 Decode 和 Prefill
M 值。

## Dispatcher 结果

| M | Latency (us) | Dispatch TFLOPS | cuBLAS TFLOPS | Dispatch/cuBLAS |
|---:|---:|---:|---:|---:|
| 1 | 54.303 | 0.618 | 1.050 | 58.9% |
| 8 | 54.999 | 4.881 | 12.153 | 40.2% |
| 16 | 56.852 | 9.443 | 9.680 | 97.6% |
| 32 | 55.910 | 19.205 | 48.100 | 39.9% |
| 64 | 69.161 | 31.051 | 109.341 | 28.4% |
| 128 | 105.779 | 40.603 | 133.577 | 30.4% |
| 512 | 329.902 | 52.076 | 163.664 | 31.8% |
| 1024 | 652.871 | 52.629 | 170.077 | 30.9% |
| 2048 | 1270.487 | 54.089 | 161.468 | 33.5% |
| 4096 | 2458.491 | 55.904 | 174.538 | 32.0% |
| 8192 | 4771.287 | 57.611 | 175.884 | 32.8% |

## 分发阈值复核

首轮 `M=64` 中，PTX MMA 为 31.325 TFLOPS，WMMA Async 为 30.954 TFLOPS。随后使用
50 warmup + 1000 measured iterations 复测：

- PTX MMA: 72.642 us，29.562 TFLOPS
- WMMA Async: 73.178 us，29.346 TFLOPS

PTX 只领先约 0.7%，不足以支持为不同 GPU 引入运行时设备查询和架构特化；`M=128`
时 WMMA Async 已领先约 7.7%。因此 sm_86/sm_89 统一保留：

```text
M <= 32 且 N、K 为 8 的倍数 -> mma_ptx
其他 Shape                    -> wmma_async
```

## 与 RTX 3060 对比

Dispatcher 在 `M=1/16/32/128/512/8192` 上相对 RTX 3060 分别约为 2.9x、2.7x、
2.8x、4.8x、5.6x、6.2x。大 Shape 的绝对吞吐由约 9.3 TFLOPS 提升到约
57.6 TFLOPS，但仍只有 cuBLAS 的约 31%~34%，后续优化重点应是更大的 CTA Tile、
更深的 `cp.async` Pipeline 和 Warp Specialization。

原始数据及图表：

- `llm_final_comparison_rtx4090.csv`
- `llm_final_rtx4090_tflops.svg`
- `llm_final_rtx4090_cublas_ratio.svg`
