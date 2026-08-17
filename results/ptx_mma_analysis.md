# PTX mma.sync / ldmatrix 阶段分析

环境：RTX 3060 (`sm_86`)、CUDA 13.3、Nsight Compute 2026.2.1。

## 实现

`mma_ptx` 不再使用 WMMA fragment API，而是显式控制 Warp 内寄存器片段：

- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 装载 Row Major A；
- `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` 装载并转置 Row Major B；
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 执行矩阵乘加；
- 按 PTX 定义的 lane group 映射直接写回四个 FP32 accumulator。

Block Tile 为 `32x32x16`，8 Warp/block，保留 16-byte `cp.async` 双缓冲。
`N/K` 不是 8 的倍数时回退到 `wmma_async`。

SASS 对应指令：

```text
LDSM.16.M88.4
LDSM.16.MT88.2
HMMA.16816.F32
```

## 性能与资源

20 次 warmup、100 次测量，单位为 TFLOPS：

| Shape | wmma_async | mma_ptx | cuBLAS | mma_ptx/cuBLAS |
|---|---:|---:|---:|---:|
| 128x128x128 | 0.759 | 0.445 | 0.923 | 48.3% |
| 512x512x512 | 6.376 | 6.629 | 10.280 | 64.5% |
| 1024x1024x1024 | 8.441 | 6.923 | 18.864 | 36.7% |
| 257x511x1025 | 3.074 | 3.044 | 8.925 | 34.1% |

完整数据见 `mma_ptx_ldmatrix.csv`。显式 PTX 在 `512^3` 略快，但在 `1024^3`
不如更大 `64x32` Tile 的 WMMA Async。原因不是寄存器压力：

| 资源 | wmma_async | mma_ptx |
|---|---:|---:|
| Registers / thread | 64 | 36 |
| Shared Memory / block | 14336 B | 4096 B |
| Achieved Occupancy | 62.21% | 95.08% |
| Grid blocks (`1024^3`) | 512 | 1024 |

较小 `32x32` Tile 产生两倍 Block，并降低 A/B 的 Block 级复用；更高 occupancy 没有
抵消同步和数据移动成本。

## Nsight 与布局实验

最终 `1024^3` profile：

- Duration：290.75 us；
- Shared load：平均 8.0 路 bank conflict；
- Barrier stall：17.9 cycles；
- Long Scoreboard：16.5 cycles；
- 无 Local Memory spill。

对 B 做按行 XOR swizzle 后冲突从 8 路降到 6 路，但普通 Benchmark 没有稳定收益；
A/B 双侧 swizzle 也未进一步降低冲突。`half2` accumulator 写回同样变慢。这些实验
证明局部 profiler 指标改善不一定带来端到端收益，因此最终版本保留无 swizzle、标量
写回的清晰基线。

最终报告见 `profile_mma_ptx_final_1024.ncu-rep`；B swizzle 与双侧 swizzle 报告分别为
`profile_mma_ptx_swizzled_1024.ncu-rep` 和
`profile_mma_ptx_ab_swizzled_1024.ncu-rep`。

片段与指令语义依据 NVIDIA PTX ISA 的 `mma.m16n8k16` 和 `ldmatrix` 章节实现。
