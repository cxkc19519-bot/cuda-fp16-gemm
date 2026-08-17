# RTX 3060 Shared Memory Profiling

测试对象：`M=N=K=1024`，RTX 3060 (`sm_86`)，CUDA 13.3，Nsight Compute
2026.2.1。报告中的 Duration 来自多 pass profiling，不与普通 benchmark 延迟混用。

## 基线定位

`register` 使用 FP16 标量读取 Shared Memory。Nsight Compute 报告：

| 指标 | register | reg_vec + padding |
|---|---:|---:|
| Nsight Duration | 861.22 us | 600.42 us |
| Shared load requests | 16,777,216 | 未触发冲突规则 |
| Shared load bank conflict | 8,388,608，平均 1.5 路 | 未检测到 |
| Warp cycles / issued instruction | 14.93 | 9.53 |
| MIO throttle | 5.4 cycles，36.5% | 未成为主要 stall |
| Registers / thread | 48 | 56 |
| Static shared memory / block | 4096 B | 4160 B |
| Theoretical / achieved occupancy | 83.33% / 68.75% | 66.67% / 58.05% |

`reg_vec` 虽然因寄存器增加而降低 occupancy，但减少 Shared Memory 指令与冲突后的收益
更大。这也是“occupancy 不是越高越快”的直接实验。

## 对照实验

1. `reg_f32sm`：把 Shared Memory 元素改成 FP32，冲突反而从平均 1.5 路升到
   2.0 路，且容量从 4096 B 增到 8192 B，性能没有提升。
2. `reg_inter`：把线程输出列由连续四列改成交错四列，shared load 仍为平均
   1.5 路，说明只改线程列编号没有解决 FP16 标量读压力。
3. `reg_vec`：转置 A tile，使 A/B 每线程所需的四个值都能用两次 `half2` 读取。
   第一个版本消除了 shared load 冲突，但转置写入产生平均 8.5 路 shared store
   冲突。
4. `reg_vec + padding`：A 的转置行 stride 从 64 改为 66 个 FP16，仅增加 64 B
   Shared Memory，同时消除转置写入冲突。

## 普通 Benchmark

20 次 warmup、100 次计时，完整数据见 `shared_vectorized_padded.csv`：

| Shape | register | reg_vec | 提升 | cuBLAS | reg_vec/cuBLAS |
|---|---:|---:|---:|---:|---:|
| 128x128x128 | 0.195 | 0.232 | 19.0% | 0.975 | 23.8% |
| 512x512x512 | 1.962 | 2.787 | 42.0% | 14.405 | 19.3% |
| 1024x1024x1024 | 2.550 | 3.651 | 43.2% | 18.948 | 19.3% |
| 257x511x1025 | 1.403 | 1.910 | 36.1% | 8.808 | 21.7% |

单位为 TFLOPS。下一阶段最明显的优化方向是改善 FP16 结果的 global store 合并；
随后进入 Tensor Core WMMA，使计算真正落到 `mma` 指令上。

## 保存的报告

- `profile_register_final_1024.ncu-rep`：最终代码下的 register 基线。
- `profile_reg_vec_1024.ncu-rep`：half2、未 padding 的中间版本。
- `profile_reg_vec_padded_1024.ncu-rep`：half2 + padding 最终版本。
- `profile_reg_f32sm_1024.ncu-rep`、`profile_reg_inter_1024.ncu-rep`：失败对照实验。
