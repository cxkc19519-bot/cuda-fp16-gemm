# Global Store Vectorization Experiment

测试环境：RTX 3060 (`sm_86`)、CUDA 13.3、Nsight Compute 2026.2.1，Shape 为
`M=N=K=1024`。

## 实现

`reg_vec4` 保留 `reg_vec` 的 Shared Memory `half2 + padding` 主循环，仅修改
epilogue：每线程把连续四个 FP16 结果打包为一个 `uint2`，执行一次 8-byte global
store。仅当地址 8-byte 对齐且四列都有效时走向量路径；任意边界 Shape 自动回退标量
写回。

## 结果

| 指标 | reg_vec | reg_vec4 |
|---|---:|---:|
| Nsight Duration | 600.42 us | 598.08 us |
| Global store sector 警告 | 仅利用 8/32 B | 未触发 |
| Warp cycles / issued instruction | 9.53 | 9.53 |
| Registers / thread | 56 | 56 |
| Shared memory / block | 4160 B | 4160 B |
| 普通 Benchmark | 3.843 TFLOPS | 3.727 TFLOPS |

向量写回改善了访存事务质量，但没有带来可重复的端到端收益。原因是每个线程在主循环
中完成大量 FMA，而结果矩阵只在 epilogue 写回一次；global store 占总执行时间很小。
因此当前最快基线继续使用 `reg_vec`，`reg_vec4` 作为 profiler 驱动但未产生性能收益的
对照版本保留。下一主线进入 Tensor Core WMMA。

完整普通计时见 `vectorized_global_store.csv`，Nsight 报告见
`profile_reg_vec4_1024.ncu-rep`。
