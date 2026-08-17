# WMMA Double Buffer / cp.async 分析

环境：RTX 3060 (`sm_86`)、CUDA 13.3、Nsight Compute 2026.2.1。基线为
`BM=64, BN=32, BK=16` 的 8-Warp `wmma_block`。

## 设计

`wmma_async` 为 A/B 各分配两套 Shared Memory Tile：

1. 启动 Tile 0 的 Global→Shared 异步拷贝并 `commit_group`；
2. `wait_group 0` 后用当前 stage 执行 WMMA；
3. Tensor Core 计算当前 stage 时，异步预取下一 K Tile 到另一个 stage；
4. 下一轮交换 stage，避免覆盖仍在使用的 Shared Tile。

主路径在 `K`、`N` 均为 8 的倍数时使用 16-byte `cp.async.cg`；其他偶数 stride
使用 4-byte `cp.async.ca`；奇数 stride 自动回退同步 `wmma_block`。异步指令的
`src-size` 操作数同时负责尾 Tile 的硬件零填充。

SASS 已确认计算与异步搬运同时存在：

```text
LDGSTS.E.BYPASS.128.ZFILL
HMMA.16816.F32
```

正确性测试额外覆盖 `65x72x40`（16-byte 异步边界）和 `65x70x34`（4-byte 异步
边界），以及原有奇数 stride 同步回退用例。

## 性能结果

20 次 warmup、100 次测量，单位为 TFLOPS：

| Shape | reg_vec | wmma_block | wmma_async | cuBLAS | async/cuBLAS |
|---|---:|---:|---:|---:|---:|
| 128x128x128 | 0.296 | 0.380 | 0.448 | 0.887 | 50.5% |
| 512x512x512 | 2.707 | 3.849 | 6.441 | 14.418 | 44.7% |
| 1024x1024x1024 | 3.407 | 4.343 | 8.117 | 17.143 | 47.3% |
| 257x511x1025 | 1.773 | 2.926 | 2.854 | 8.536 | 33.4% |

奇数 stride Shape 走同步回退，因此最后一行不期待异步收益。完整数据见
`wmma_cp_async.csv`。

## Nsight 对比（1024³）

| 指标 | wmma_block | wmma_async |
|---|---:|---:|
| Profile Duration | 468.42 us | 245.66 us |
| Memory Throughput | 47.79% | 83.90% |
| Long Scoreboard | 6.6 cycles，31.9% | 不再是主要 stall |
| Registers / thread | 40 | 64 |
| Shared Memory / block | 11264 B | 14336 B |
| Achieved Occupancy | 90.91% | 62.21% |
| Local Memory spill | 0 | 0 |

虽然双缓冲增加了寄存器和 Shared Memory、降低 occupancy，但隐藏数据搬运后的收益
明显更大。当前主要 stall 转为 CTA barrier（7.1 cycles，35.3%），shared fragment
load 仍有平均 11.6 路 bank conflict。后续优化应考虑更底层的 `mma.sync`/`ldmatrix`
布局和 Warp specialization，而不是单纯追求更高 occupancy。

最终报告见 `profile_wmma_async_final_1024.ncu-rep`。
