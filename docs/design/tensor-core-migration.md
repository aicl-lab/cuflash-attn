# Tensor Core 迁移计划

> 状态：**Phase 2（前向）已实现** —— FP16/BF16 的 WMMA 前向 kernel 已在 v0.5.0 落地（`src/forward/flash_attention_forward_wmma.cu`），带运行时分派，标量路径保留为 fallback。单元测试套件（`ctest`）在本机 sm_86（RTX 3060）上通过；真实硬件的数值对比、benchmark 与 compute-sanitizer 结果见 `docs/performance/validation-v1.0.md`。**Backward 的 WMMA 化明确不纳入 v1.0**（保持 scalar），后续阶段另议。每个阶段都设计为可在真实硬件上独立落地并验证。

## 为什么需要它

CuFlash-Attn 正确但慢。kernel 把每个 matmul 都写成朴素标量循环——一个线程产出一个输出元素、
对 `K`/`HEAD_DIM` 维串行累加（`src/kernels/impl/tile_io.cuh` 的 `matmul_ABt`），而融合的
softmax/`P@V` 段更是**一个线程包办一整行 query**（`src/forward/flash_attention_forward_typed.cu`）。
没有 Tensor Core、没有 `cp.async`、没有寄存器分块、没有 warp 级协作。

对一个低精度注意力 kernel 而言，约 90% 的可达吞吐来自 Tensor Core。本文档把这条路径拆成
若干阶段，每个阶段都先编译通过、跑通现有测试、并看到可测量的加速，再进入下一阶段。

**每个阶段的铁律**：在现有测试（`test_forward`、`test_backward`、`test_dtype`、PyTorch 对比）
和 `scripts/run_compute_sanitizer.sh` 全部通过之后再往前走。不要叠加未验证的改动。

## 阶段 0 —— 基线与测量（先做这个）

优化之前，先让差距可见、可复现：

1. 在目标 GPU 上跑 `benchmarks/bench_flash_attention.cu`（现已报告 `TFLOP/s` 与 `HBM GB/s`），
   把数字连同确切的设备/驱动/CUDA 版本记录到 `docs/performance/benchmarks.md`。
2. 加入朴素 materialized 基线（benchmark 里已有）以及相同形状下官方 `flash-attn` / PyTorch SDPA 的数字。
3. 用 Nsight Compute 剖析一次前向 launch，记录实测 occupancy、SM 吞吐、内存吞吐——
   判断阶段 1 和阶段 2 哪个收益更大。

退出标准：一份按形状记录的基线 TFLOP/s 表格入库。

## 阶段 1 —— 寄存器分块与 warp 协作（不用 Tensor Core）

最便宜的大收益，也为后续一切去风险。把 "一线程一行" 的内层循环替换为分块方案：每个线程拥有一小块
`MxN` 的输出寄存器块，整个 block 按 warp 协作。

- shared memory 仍保留 FP32，但让每个线程累加一个 `TM x TN` 微块（如 `4 x 4`）的 `S` 和 `O`，
  把 `Q`/`K`/`V` 读进寄存器。
- 仅这一步通常就能比标量版快数倍（降低指令开销与 shared memory 流量），并且它正是 Tensor Core
  阶段要复用的脚手架。
- 对 shared memory 行做 padding（或 swizzle 索引）以消除 bank conflict——当前
  `A[row*K+k]` / `B[col*K+k]` 的访问模式在 `K` 为 32 的倍数时正存在这个问题。

退出标准：前向+反向通过测试与 compute-sanitizer；benchmark 显示加速；数字入库。

## 阶段 2 —— 为 FP16/BF16 引入 WMMA（sm_70+）

通过 `nvcuda::wmma` API 为低精度路径引入 Tensor Core（FP32 留在阶段 1 的 CUDA core 路径）。

- 把 `Q`/`K`/`V` tile **以输入精度**（half / bf16）存放在 shared memory，而不是上转为 float——
  这能把 shared memory 压力减半，也是 block 能变大的前提。
- 用 `wmma::fragment`（half 用 `m16n16k16`；bf16 fragment 需 sm_80+）计算 `S = Q @ Kᵀ` 与
  `O += P @ V`，累加到 `float` fragment。
- online softmax 状态（`m`、`l`）与延迟归一化保持现状，只替换两个 matmul。
- 反向的三个 matmul（`dV = Pᵀ@dO`、`dP = dO@Vᵀ`、`dQ/dK = dS·…`）同样处理。

`QKᵀ` 步骤示意：

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;   // Q tile
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;   // K tile（col_major 即 Kᵀ）
fragment<accumulator, 16, 16, 16, float>        c_frag;   // S tile
fill_fragment(c_frag, 0.0f);

// 每个 warp 拥有一个 16x16 的 S 子块；沿 K 维以 16 为步长循环
for (int k0 = 0; k0 < HEAD_DIM; k0 += 16) {
    load_matrix_sync(a_frag, Q_smem + row0 * HEAD_DIM + k0, HEAD_DIM);
    load_matrix_sync(b_frag, K_smem + col0 * HEAD_DIM + k0, HEAD_DIM);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
}
// c_frag 现在是该 warp 16x16 块的 scale * S；接着施加 scale 与 softmax
```

退出标准：FP16/BF16 前向+反向在现有容差内匹配 FP32 参考；compute-sanitizer 干净；
benchmark 在 `head_dim ∈ {64,128}` 上相对阶段 1 显示大幅倍数提升。

## 阶段 3 —— 用 `cp.async` 重叠加载与计算（sm_80+）

通过对 K/V tile 双缓冲来隐藏全局内存延迟：

- 用 `cuda::memcpy_async` / `cp.async` 把下一个 K/V tile 预取到第二块 shared memory，
  同时消费当前 tile。
- 用 `cuda::pipeline` barrier 协调，取代整 block 的 `__syncthreads`。
- 这里结合 opt-in shared memory 上限（`prepare_dynamic_smem_launch` 已接通）调优 block 大小与 occupancy。

退出标准：Nsight Compute 中 SM 吞吐更高；benchmark 提升；测试与 sanitizer 全绿。

## 阶段 4 —— Hopper：WGMMA + TMA（sm_90，可选）

要达到 H100 级吞吐，需转向 warpgroup MMA（`wgmma`）与用于异步批量拷贝的 Tensor Memory Accelerator，
即 FlashAttention-3 风格。这是最大的工程，应在阶段 1–3 稳固并测量之后再开始。
此处建议依赖 CUTLASS，而非手写 PTX。

## 横切关注点

- **运行时按架构分发。** 保留标量/阶段 1 路径作为任何缺少所需 Tensor Core 指令架构的 fallback；
  在 `launch_flash_attention_forward_typed` 里通过编译期架构检查选择 kernel。
- **`head_dim` 覆盖。** WMMA fragment 宽 16；`head_dim ∈ {32,64,128}` 都能整除。保留现有
  `is_supported_head_dim` 门禁。
- **数值一致性。** 累加全程保持 FP32；唯一的精度变化是 shared memory tile 的存储格式，
  而这正是现有 FP16/BF16-vs-FP32 测试所守护的。
- **L 保持 FP32。** logsumexp 输出为 `float*`（见 CHANGELOG）；重构 shared memory 时不要回退这一点。

## 怎样算 "完成"

README 的性能表可在某台具名设备上由 `benchmarks/` 复现，CuFlash-Attn 在 `head_dim=128` 上与官方
FlashAttention 相差在较小倍数内，且每个数字都由入库的 benchmark 运行支撑，而非估算。
