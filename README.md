# CUDA FP16 GEMM

从零实现并逐步优化面向 LLM 推理 Shape 的 CUDA FP16 GEMM。当前完成：

- Row-major `C[M,N] = A[M,K] * B[K,N]`
- FP16 输入、FP32 累加、FP16 输出
- CPU 与 cuBLAS 正确性基线、cuBLAS 性能基线
- Naive CUDA Core Kernel
- Coalesced Memory Access Kernel
- Shared Memory Tiled Kernel
- Register Tiling 与 `half2` Shared Memory 向量化
- Tensor Core WMMA 与多 Warp Block Tiling
- Ampere `cp.async` 双缓冲流水线
- 显式 PTX `ldmatrix` / `mma.sync` Kernel
- LLM Decode/Prefill Shape Suite 与 Shape-Aware Dispatch
- 非整数边界 Shape 正确性测试
- CUDA Event 延迟测量、TFLOPS 和相对 cuBLAS 性能

## 当前目标硬件

- 本地：RTX 3060，Compute Capability 8.6
- 服务器：RTX 4090，Compute Capability 8.9

CMake 默认生成 `sm_86` 代码。服务器构建时通过参数切换到 `sm_89`，源码不写死 GPU 型号。

## 依赖

- 支持 C++17 的主机编译器
- CUDA Toolkit（包含 cuBLAS）
- CMake 3.22+

Windows 推荐安装 Visual Studio 2022 的“使用 C++ 的桌面开发”组件，并安装 CUDA Toolkit。

## 构建

RTX 3060：

```powershell
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release -j
```

RTX 4090：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

## 正确性测试

首先使用 `3x5x7` 的非方阵验证 CPU reference 与 Row Major cuBLAS wrapper，
再使用 cuBLAS FP16 输入、FP32 compute、FP16 输出检查各个 GPU Kernel：

```powershell
ctest --test-dir build -C Release --output-on-failure
```

测试包括方阵、非方阵、尺寸小于单个 Tile 的 `17x19x23`，以及
`M=257, N=511, K=1025` 的非 Tile 整数边界 Shape。另有 `65x72x40` 和
`65x70x34` 专门覆盖 16-byte/4-byte `cp.async` 的边界零填充路径。所有 Kernel
使用同一组输入和同一个 cuBLAS reference。

cuBLAS 默认按 Column Major 解释数据。本项目的 wrapper 利用
`C^T = B^T * A^T`，交换 A/B 的调用位置，使公开接口始终保持 Row Major。

## Benchmark

```powershell
.\build\Release\gemm_benchmark.exe --warmup 20 --iterations 100 --csv results\phase1.csv
```

Linux/单配置生成器下通常为：

```bash
./build/gemm_benchmark --warmup 20 --iterations 100 --csv results/phase1.csv
```

只运行单个 Kernel/Shape，适合 Nsight Compute：

```powershell
.\build\gemm_benchmark.exe --kernel register --shape 1024 1024 1024 --warmup 0 --iterations 1
```

任务书中的 LLM Shape 可直接成组运行（固定 `N=K=4096`）：

```powershell
.\build\gemm_benchmark.exe --suite llm-decode --kernel dispatch --csv results\llm_decode.csv
.\build\gemm_benchmark.exe --suite llm-prefill --kernel dispatch --csv results\llm_prefill.csv
.\build\gemm_benchmark.exe --suite llm-all --kernel dispatch --csv results\llm_all.csv
```

Decode 覆盖 `M={1,2,4,8,16,32,64,128}`，Prefill 覆盖
`M={128,512,1024,2048,4096,8192}`。`--kernel dispatch-candidates` 用于同场比较
参与分发决策的候选内核。

最终对比和绘图可通过脚本复现：

```powershell
python scripts\run_benchmark.py --warmup 20 --iterations 100 `
  --output results\llm_final_comparison.csv
python scripts\plot_results.py results\llm_final_comparison.csv
```

`final-comparison` 组包含 RegVec、WMMA Block、PTX MMA、WMMA Async、Dispatcher 和
cuBLAS，主动排除在大 LLM Shape 上运行时间过长的教学型 Naive Kernel。绘图脚本仅依赖
Python 标准库，输出可直接在浏览器和 Markdown 中查看的 SVG。

当前测试：

- `128 x 128 x 128`
- `512 x 512 x 512`
- `1024 x 1024 x 1024`
- `257 x 511 x 1025`

输出字段：

```text
kernel,M,N,K,latency_us,tflops,cublas_ratio
```

其中：

```text
FLOPs  = 2 * M * N * K
TFLOPS = FLOPs / (latency_us * 1e6)
```

Benchmark 只计 GPU 工作，不包含显存分配、随机数据生成和 Host/Device 数据拷贝。

## 当前实现版本

V0 到 V2 使用一个线程计算一个输出元素；V3 开始让每线程计算一个输出子块。

### V0 Naive

V0 使用最直接的 `threadIdx.x -> row` 映射：

```text
thread(row, col) -> C[row, col]
```

由于 Row Major 的连续维度是 column，同一个 Warp 的相邻线程会跨行访问 C，形成较大的地址步长。

### V1 Coalesced

V1 只修改线程映射：

```text
threadIdx.x -> col
threadIdx.y -> row
```

CUDA 在线性化二维 Block 时 x 维变化最快，因此同一 Warp 对 B 和 C 的访问尽量落在连续地址上。
V1 没有减少理论读取次数，用于单独观察合并访存的收益。

### V2 Shared Memory Tiling

V2 保留两种教学型配置做对照实验：

```text
shared32: BM = BN = BK = 32, 1024 threads/block
shared16: BM = BN = BK = 16,  256 threads/block
```

一个 Block 协同把 A/B Tile 从 Global Memory 搬到 Shared Memory，再重复使用它们计算 C Tile。
每个 K Tile 在计算前后各执行一次 `__syncthreads()`；非完整边界 Tile 使用 0 填充。

`shared32` 数据复用更高，但 1024 threads/block 限制每个 SM 的驻留 Block 数；`shared16`
复用较低，但调度更灵活。项目保留两者并用实测选择后续 Register Tiling 的基线。

### V3 Register Tiling

V3 使用：

```text
BM = 64, BN = 64, BK = 16
TM = 4,  TN = 4
256 threads/block
```

每个线程不再只计算一个元素，而是在寄存器中保存 `4x4` 个 FP32 accumulator。每个 K
位置从 Shared Memory 读取 4 个 A 和 4 个 B，随后通过外积产生 16 次 FMA，从而提高
Shared Memory 数据的线程级复用率。

为验证 FP16 Shared Memory 与 4-byte bank 粒度的关系，另保留 `reg_f32sm` 实验版本：
Global Memory 输入仍为 FP16，但加载 Tile 时转换为 FP32 存入 Shared Memory。该版本不是
凭空替换基线，而是用于对比 Bank Conflict、MIO Throttle 与 Shared Memory 容量开销。

`reg_inter` 进一步改变每线程 4 个输出列的分配方式。原版本让线程 x 负责
`[4x, 4x+1, 4x+2, 4x+3]`；交错版改为 `[x, x+16, x+32, x+48]`。因此在固定
thread-tile column 上，Warp 相邻线程访问连续的 B Shared Memory 地址，同时最终 C
写回也保持连续。

`reg_vec` 针对剖析中暴露的 Shared Memory 标量读压力：A tile 以转置布局存入
Shared Memory，使每线程所需的 4 个 A 值连续；A/B 两侧随后都用两次 `half2` 读取
替代四次 FP16 标量读取。转置 A tile 的 stride 额外 padding 两个 FP16，以避免转置
写入的 bank conflict。

`reg_vec4` 是 Global Store 对照实验：每线程把连续四个 FP16 结果打包成一次 8-byte
写回，边界或未对齐地址自动回退标量路径。Nsight 中原先“每 sector 仅利用 8/32 B”
的警告被消除，但 `1024^3` profile Duration 仅从 600.42 us 变为 598.08 us，普通
Benchmark 也没有稳定收益。说明输出只写一次，当前 GEMM 的主导成本仍在计算主循环；
因此 `reg_vec` 保持为最快 CUDA Core 基线。详细记录见
`results/global_store_analysis.md`。

### V4 Tensor Core WMMA

`wmma` 是单 Warp 教学基线：一个 Warp 使用 `16x16x16` WMMA fragment 计算一个
输出 Tile。SASS 已确认生成 `HMMA.16816.F32`，但每个输出 Tile 独立搬运 A/B，
`1024^3` 只有 1.773 TFLOPS。

`wmma_block` 改为 `BM=64, BN=32, BK=16`，8 个 Warp 在一个 Block 内协作。A Tile
被两个 Warp 复用，B Tile 被四个 Warp 复用，`1024^3` 达到 4.403 TFLOPS。另保留
`wmma_64` 和 Shared Tile padding 两组对照实验；更大的 Block 或单独消除 bank
conflict 都因资源/occupancy 代价而变慢。详细分析见 `results/wmma_analysis.md`。

### V5 Double Buffer / cp.async

`wmma_async` 为 A/B 各准备两套 Shared Tile，使用 Ampere `cp.async` 在 Tensor Core
计算当前 K Tile 时预取下一 Tile。对齐主路径使用 16-byte 事务，偶数非 8 倍数 stride
使用 4-byte 事务，奇数 stride 自动回退同步 WMMA。SASS 已确认同时生成
`LDGSTS.E.BYPASS.128.ZFILL` 和 `HMMA.16816.F32`。

在 `1024^3` 上，异步版本从同步 WMMA 的 4.343 提升到 8.117 TFLOPS，达到本轮
cuBLAS 的 47.3%。Nsight Duration 从 468.42 us 降到 245.66 us，Long Scoreboard
不再是主要 stall。详细分析见 `results/cp_async_analysis.md`。

### V6 PTX ldmatrix / mma.sync

`mma_ptx` 显式使用 `ldmatrix.x4` 装载 A、`ldmatrix.x2.trans` 装载 B，并执行
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`。Block Tile 为 `32x32x16`，
8 Warp，并保留 16-byte `cp.async` 双缓冲。SASS 对应为 `LDSM.16.M88.4`、
`LDSM.16.MT88.2` 与 `HMMA.16816.F32`。

该版本把资源降到 36 registers/thread、4096 B Shared Memory，`512^3` 达到
6.629 TFLOPS，略快于 WMMA Async；但 `1024^3` 因 Tile 更小、Block 数和同步成本
更高，只有 6.923 TFLOPS，未替换 8.441 TFLOPS 的 WMMA Async 大 Shape 基线。
详细分析见 `results/ptx_mma_analysis.md`。

### V7 LLM Shape-Aware Dispatch

`dispatch` 将 Shape 选择和 Kernel 实现分离。RTX 3060、`N=K=4096` 的 Decode 实测
显示，`mma_ptx` 在 `M=1~32` 始终是最快自研版本；从 `M=64` 起，Tile 更大且流水线
吞吐更高的 `wmma_async` 反超。因此 sm_86 的当前策略为：

```text
M <= 32 且 N、K 均为 8 的倍数 -> mma_ptx
其他 Shape                         -> wmma_async
```

对齐限制确保 PTX 路径使用 16-byte `cp.async`；通用路径继续覆盖偶数 4-byte copy 和
奇数 stride 的同步回退。阈值选择、候选数据和完整 LLM 结果见
`results/llm_shape_dispatch_analysis.md`。

### V8 Final Benchmark / Visualization

自动化脚本在完整 LLM Shape 上统一测量六条性能曲线，并生成 TFLOPS 与相对 cuBLAS
两张图。本机最终对比中，Dispatcher 在 `M=1/16/32` 分别达到 cuBLAS 的
85.6%/91.6%/101.5%；Prefill 区域稳定在约 9.3 TFLOPS。完整结果与局限分析见
`results/final_benchmark_analysis.md`。

### V9 RTX 4090 / sm_89 Validation

服务器使用 CUDA 12.4、CMake 3.22 和两张 RTX 4090。项目以
`-DCMAKE_CUDA_ARCHITECTURES=89` 完成 Release 构建，全部正确性测试通过。完整 LLM
Shape 上，Dispatcher 从 `M=1` 的 0.618 TFLOPS 提升到 `M=8192` 的 57.611 TFLOPS；
`M=16` 达到 cuBLAS 的 97.6%。`M=64` 的 PTX/WMMA Async 1000 次复测仅相差约 0.7%，
而 `M=128` 起 WMMA Async 明确领先，因此继续保留稳健的 `M<=32` 分发阈值。详见
`results/rtx4090_analysis.md`。

#### RTX 4090 FP16 GEMM 性能汇总

与 RTX 3060 汇总表相同的四个 Shape，在 RTX 4090 上实测如下（单位：TFLOPS）：

| Shape | RegVec | WMMA Block | WMMA Async | MMA PTX | cuBLAS | Best/cuBLAS |
|---|---:|---:|---:|---:|---:|---:|
| `128x128x128` | 0.415 | 0.502 | 1.006 | 1.206 | 0.785 | 153.7% |
| `512x512x512` | 7.652 | 9.620 | 24.848 | 25.095 | 43.330 | 57.9% |
| `1024x1024x1024` | 21.098 | 27.951 | 51.641 | 34.190 | 105.607 | 48.9% |
| `257x511x1025` | 3.868 | 4.933 | 4.932 | 4.933 | 24.098 | 20.5% |

原始汇总数据为 `results/rtx4090_gemm_summary.csv`。

#### RTX 3060 初步结果

环境：RTX 3060 12GB、`sm_86`、CUDA 13.3、Release build。每项执行 20 次 warmup 和
100 次正式测量。以下数据是本机实测值，不代表其他 GPU 或温度/频率状态下的结果。

| Shape | RegVec | WMMA Block | WMMA Async | MMA PTX | cuBLAS | Best/cuBLAS |
|---|---:|---:|---:|---:|---:|---:|
| `128x128x128` | 0.295 | 0.384 | 0.759 | 0.445 | 0.923 | 82.2% |
| `512x512x512` | 2.695 | 4.190 | 6.376 | 6.629 | 10.280 | 64.5% |
| `1024x1024x1024` | 3.454 | 4.426 | 8.441 | 6.923 | 18.864 | 44.7% |
| `257x511x1025` | 1.881 | 3.005 | 3.074 | 3.044 | 8.925 | 34.4% |

单位均为 TFLOPS。完整数据保存在 `results/mma_ptx_ldmatrix.csv`。

结论：

- Coalesced 在大方阵上相对 Naive 提升约 4.3 倍。
- `shared16` 比 `shared32` 更快，但一线程一输出的两个 Shared 版本仍未超过 Coalesced。
- Register Tiling 在 `1024^3` 上相对 Coalesced 提升约 3.1 倍，证明线程级数据复用有效。
- `half2` Shared Memory 向量化配合 A tile padding，把 `1024^3` 从 2.550 提升到
  3.651 TFLOPS，相对标量 Register 版本提升 43.2%。
- Register Tiling 在 `128^3` 上更慢，因为 `64x64` Block Tile 只产生 4 个 Block，GPU
  并行度不足。这为后续 Shape-Aware Dispatch 提供了直接实验依据。
- 多 Warp WMMA 在 `1024^3` 上相对本轮 `reg_vec` 提升 26.8%，相对单 Warp WMMA
  提升 2.48 倍，证明 Tensor Core 需要与 Block 级数据复用共同设计。
- 16-byte `cp.async` 双缓冲让 `1024^3` 相对同步 WMMA 再提升 86.9%，也再次说明
  occupancy 从 90.9% 降到 62.2% 并不等于性能下降。

最终代码下，`cuobjdump --dump-resource-usage` 显示标量 Register Kernel 使用 48
registers/thread、4096 B Shared Memory；`reg_vec` 使用 56 registers/thread、4160 B
Shared Memory，二者都没有 Local Memory spill。

Nsight Compute 在 `1024^3` 上显示：标量版有 8,388,608 次 shared load bank conflict，
平均 1.5 路，MIO throttle 为 5.4 cycles；`half2 + padding` 版没有触发 shared
load/store conflict 规则，Warp cycles per issued instruction 从 14.93 降到 9.53，profile
Duration 从 861.22 us 降到 600.42 us。详细证据链见 `results/nsight_analysis.md`。

## 优化路线

后续版本会独立保留：

1. Warp Specialization 与同步优化
2. LLM Shape benchmark 与 Shape-Aware Dispatch
3. RTX 4090 (`sm_89`) 复测与参数调优

每个版本遵循：正确性测试 → Benchmark → Nsight Compute 分析 → 下一项优化。
