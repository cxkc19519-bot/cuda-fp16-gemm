# Tensor Core WMMA 阶段分析

环境：RTX 3060 (`sm_86`)、CUDA 13.3、Nsight Compute 2026.2.1。所有实现保持
Row Major `C=A*B`、FP16 输入、FP32 累加、FP16 输出，并支持非 16 倍数边界。

## 版本演进

### wmma：单 Warp 基线

一个 Warp 计算一个 `16x16x16` Tile。每个 K Tile 都先把 A/B 搬入 Shared Memory，
尾块零填充。SASS 中已确认生成：

```text
HMMA.16816.F32
```

`1024^3` 只有 1.773 TFLOPS。Tensor Core 指令本身不是充分条件：单 Warp Block
无法跨输出 Tile 复用 A/B，搬运和同步成本超过计算收益。

### wmma_block：8 Warp 协作

采用 `BM=64, BN=32, BK=16`，256 threads/block。8 个 Warp 排成 `4x2`：每份
A Tile 被 N 方向两个 Warp 复用，每份 B Tile 被 M 方向四个 Warp 复用。

`1024^3` 达到 4.403 TFLOPS，相对单 Warp WMMA 提升 2.48 倍，相对本轮
`reg_vec` 的 3.473 TFLOPS 提升 26.8%。资源使用为 40 registers/thread、11264 B
Shared Memory/block，无 Local Memory spill。

### 对照实验

- Shared Tile 行 padding 8 个 FP16：消除平均 11.5 路 shared load bank conflict，
  但 registers/thread 从 40 增到 52，achieved occupancy 从 90.8% 降到 61.0%，
  性能下降到约 3.53 TFLOPS，因此未作为最终版本。
- `wmma_64`：`64x64x16`、16 Warp、512 threads/block，让 A/B 都复用四次；但更大
  Block 和 accumulator staging 降低调度弹性，`1024^3` 为 3.741 TFLOPS，仍慢于
  `64x32` 版本。

## 最终普通 Benchmark

20 次 warmup、100 次测量，单位为 TFLOPS：

| Shape | reg_vec | wmma | wmma_block | wmma_64 | cuBLAS | wmma_block/cuBLAS |
|---|---:|---:|---:|---:|---:|---:|
| 128x128x128 | 0.301 | 0.310 | 0.385 | 0.322 | 0.862 | 44.6% |
| 512x512x512 | 2.772 | 1.961 | 4.195 | 3.554 | 11.534 | 36.4% |
| 1024x1024x1024 | 3.473 | 1.773 | 4.403 | 3.741 | 17.971 | 24.5% |
| 257x511x1025 | 1.892 | 1.477 | 2.944 | 2.551 | 8.247 | 35.7% |

完整数据见 `wmma_block_tiling.csv`。

## 最终 Nsight 结论

`wmma_block` 在 `1024^3` 上：

- profile Duration：468.42 us；
- theoretical / achieved occupancy：100% / 90.91%；
- 40 registers/thread、11264 B Shared Memory/block、无 spill；
- Compute Throughput：54.75%，Issue Slots Busy：48.14%；
- shared fragment load 仍有平均 11.5 路冲突；
- Long Scoreboard：6.6 cycles，占 Warp 周期 31.9%。

下一阶段应通过双缓冲或 `cp.async` 重叠 Global→Shared 搬运与 Tensor Core 计算，
而不是继续增加 occupancy。最终报告见 `profile_wmma_block_final_1024.ncu-rep`。
