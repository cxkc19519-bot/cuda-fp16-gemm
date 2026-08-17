# LLM Shape Benchmark 与 Dispatch 分析（RTX 3060）

## 实验设置

- GPU: NVIDIA GeForce RTX 3060, sm_86
- CUDA: 13.3, Release build
- 数据类型: FP16 input/output, FP32 accumulation
- Shape: N=K=4096
- 候选测量: 10 warmup + 30 iterations
- Dispatcher 复测: 20 warmup + 100 iterations

Benchmark 只统计 CUDA Event 包围的 GPU 工作，不计随机数据、分配和 H2D copy。

## Decode 候选结论

| M | Coalesced | RegVec | WMMA Block | WMMA Async | MMA PTX | cuBLAS | 最快自研 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.049 | 0.040 | 0.072 | 0.109 | 0.212 | 0.248 | MMA PTX |
| 8 | 0.374 | 0.359 | 0.599 | 0.923 | 1.776 | 1.926 | MMA PTX |
| 16 | 0.669 | 0.712 | 1.169 | 1.859 | 3.505 | 3.844 | MMA PTX |
| 32 | 0.654 | 1.330 | 2.270 | 3.303 | 6.746 | 6.163 | MMA PTX |
| 64 | 0.725 | 2.522 | 4.042 | 6.434 | 6.991 | 15.527 | 边界，需复测* |
| 128 | 0.756 | 3.376 | 4.458 | 8.561 | 6.951 | 20.263 | WMMA Async |

单位为 TFLOPS。M=64 的首轮候选测量中 MMA PTX 高约 8.7%，因此额外以 20 warmup、
100 iterations 对两个内核单独复测；WMMA Async 为 6.539 TFLOPS，MMA PTX 为
6.521 TFLOPS，差距缩小至约 0.3% 并发生反转。结合 M=128 以后 WMMA Async 的稳定
优势，阈值取在 32/64 边界，避免依据边界噪声把大 M 分配给吞吐平台较低的 32x32
PTX Tile。

## Prefill 结论

WMMA Async 在 M=512、1024、2048、4096、8192 上分别达到 9.227、9.045、9.329、
9.087、9.336 TFLOPS。MMA PTX 的对应候选结果约为 7.18~7.49 TFLOPS，因此 Prefill
统一选择 WMMA Async。

## 最终策略

```text
if M <= 32 and N % 8 == 0 and K % 8 == 0:
    mma_ptx
else:
    wmma_async
```

`N/K % 8` 条件不是性能猜测，而是显式 PTX Kernel 16-byte copy 主路径的布局约束。
策略由 `select_gemm_kernel()` 暴露，可在 RTX 4090 上复测后独立调整。

完整原始数据：

- `llm_decode_candidates_rtx3060.csv`
- `llm_prefill_wmma_async_rtx3060.csv`
- `llm_prefill_mma_ptx_rtx3060.csv`
- `llm_dispatch_rtx3060.csv`
