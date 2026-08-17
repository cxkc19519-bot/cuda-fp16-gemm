# 最终 LLM Shape 性能对比（RTX 3060）

## 配置

- GPU: NVIDIA GeForce RTX 3060, sm_86
- CUDA: 13.3, Release build
- Shape: M=1~8192, N=K=4096
- 10 warmup + 30 measured iterations
- FP16 input/output, FP32 accumulation

参与最终对比的版本为 RegVec、WMMA Block、PTX MMA、WMMA Async、Shape-Aware
Dispatcher 和 cuBLAS。Naive 等教学内核没有放入最终大 Shape 长测。

## 核心结果

| M | Dispatch TFLOPS | cuBLAS TFLOPS | Dispatch/cuBLAS |
|---:|---:|---:|---:|
| 1 | 0.213 | 0.248 | 85.6% |
| 8 | 1.760 | 1.892 | 93.0% |
| 16 | 3.509 | 3.831 | 91.6% |
| 32 | 6.766 | 6.669 | 101.5% |
| 64 | 7.008 | 15.531 | 45.1% |
| 128 | 8.531 | 20.217 | 42.2% |
| 512 | 9.337 | 25.388 | 36.8% |
| 2048 | 9.368 | 25.567 | 36.6% |
| 8192 | 9.264 | 26.513 | 34.9% |

## 结论

1. Dispatcher 几乎与被选 Kernel 重合，普通 host-side 分支没有可测的额外 launch 开销。
2. 小 M Decode 是当前项目最有竞争力的区域。32x32 PTX Tile 降低了无效行计算，
   `M=8~32` 达到约 92%~102% cuBLAS。
3. `M=64` 以后 cuBLAS 吞吐快速上升，而当前 WMMA Async 平台约为 9.3 TFLOPS，说明
   大 Shape 的主要后续空间在更大的 CTA Tile、更深流水和 Warp Specialization，而不在
   dispatcher 条件本身。
4. `M=8192` 上，RegVec、WMMA Block、PTX MMA、WMMA Async 分别为 3.995、5.305、
   7.322、9.319 TFLOPS，完整展示了 CUDA Core 到 Tensor Core 再到异步流水的优化收益。

原始数据为 `llm_final_comparison_rtx3060.csv`，对应图片为
`llm_final_rtx3060_tflops.svg` 和 `llm_final_rtx3060_cublas_ratio.svg`。

## RTX 4090 复测

```bash
cmake -S . -B build-sm89 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-sm89 -j
python scripts/run_benchmark.py --binary build-sm89/gemm_benchmark \
  --warmup 20 --iterations 100 --output results/llm_final_comparison_rtx4090.csv
python scripts/plot_results.py results/llm_final_comparison_rtx4090.csv \
  --prefix llm_final_rtx4090 --title "FP16 GEMM on RTX 4090: LLM Shape Comparison"
```

sm_89 结果产生后应重新检查 PTX/WMMA 在 M=32、64、128 的交叉点，不直接沿用 sm_86
阈值作为最终结论。
