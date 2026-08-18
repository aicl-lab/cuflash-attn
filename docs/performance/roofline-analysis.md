# Roofline 分析

> **版本**: v1.0（HBM 流量模型已按本实现实际结构修正：固定 tile、K/V 按 Q block 重载；WMMA 路径见 tensor-core 迁移文档）  
> **适用范围**: CuFlash-Attn 前向/反向 kernel，FP16，causal/non-causal  
> **前置阅读**: [基准测试](./benchmarks.md)（含实测带宽与耗时数据）

---

## 1. Roofline 模型简介

Roofline 模型是一种**面向吞吐量**的性能分析框架，它将算法性能受限于两个互斥资源：

1. **内存带宽（Bandwidth Roof）**——单位时间内可从 HBM（High Bandwidth Memory）读写数据的最大字节数，记为 $\beta_{peak}$（GB/s）。
2. **峰值算力（Compute Roof）**——单位时间内 Tensor Core / CUDA Core 可完成的浮点运算数，记为 $\pi_{peak}$（TFLOPS）。

算法在这两个极限之间处于哪种 regime，由其**算术强度（Arithmetic Intensity, AI）**决定：

$$
AI = \frac{\text{总浮点运算数 (FLOPs)}}{\text{总 HBM 访存量 (Bytes)}}
$$

单位为 $\text{FLOP}/\text{Byte}$。Roofline 的"屋顶"形状为分段函数：

$$
P_{roofline}(AI) = \min(\pi_{peak},\; \beta_{peak} \times AI)
$$

其几何意义如下：

| 概念 | 定义 | 图示位置 |
|------|------|---------|
| **Memory-bound regime** | $AI < AI_{ridge}$，性能由斜率为 $\beta_{peak}$ 的直线限制 | Roofline 左侧斜线区域 |
| **Compute-bound regime** | $AI > AI_{ridge}$，性能由水平线 $\pi_{peak}$ 限制 | Roofline 右侧平顶区域 |
| **Ridge Point（脊点）** | $AI_{ridge} = \pi_{peak} / \beta_{peak}$，带宽与算力限制的交界 | 斜线与水平线的交点 |

> **工程直觉**: 若算法位于 ridge point 左侧，再增加 Tensor Core 算力也**无济于事**；必须减少 HBM 流量或提高 $AI$。FlashAttention 的核心价值正是通过 tiling 与在线 softmax 将 Attention 从 ridge point 的极左侧向右推移，但仍处于 memory-bound 区间。

---

## 2. 目标 GPU 的理论峰值

以下数值均为厂商标称的**dense FP16 Tensor Core**峰值，非稀疏、非低精度（INT8/FP8）。

| GPU | 架构 | HBM 带宽 $\beta_{peak}$ | FP16 算力 $\pi_{peak}$ | Ridge Point $AI_{ridge}$ | TDP |
|:---|:---|:---:|:---:|:---:|:---:|
| NVIDIA V100 | Volta (`sm_70`) | 900 GB/s | 31.4 TFLOPS | 34.9 FLOP/Byte | 300 W |
| NVIDIA A100 | Ampere (`sm_80`) | 2,039 GB/s | 312 TFLOPS | 153 FLOP/Byte | 400 W |
| NVIDIA H100 | Hopper (`sm_90`) | 3,350 GB/s | 989 TFLOPS | 295 FLOP/Byte | 700 W |

### 2.1 Ridge Point 的工程含义

| GPU | Ridge Point 解读 |
|:---|:---|
| V100 | 每从 HBM 读取 1 Byte，必须至少做 35 次 FP16 运算才能"回本"进入 compute-bound。否则性能被带宽锁死。 |
| A100 | Ampere 的 Tensor Core 算力提升近 10×，但带宽仅提升 2.3×，ridge point 大幅右移至 153。这意味着大量传统 kernel（GEMM 以外的）在 A100 上更容易落入 memory-bound。 |
| H100 | Hopper 的 ridge point 达到 295。FlashAttention-3 引入的 TMA + WGMMA 本质上是**在硬件层面进一步减少 HBM 流量**，从而将有效 $AI$ 向右推移，逼近 ridge point。 |

---

## 3. FlashAttention 算术强度推导

### 3.1 标准 Attention 的计算与访存

对于输入 $Q, K, V \in \mathbb{R}^{B \times H \times N \times d}$，标准 Attention（无 tiling，materialize 中间矩阵）的计算流程为：

$$
S = QK^T \in \mathbb{R}^{B \times H \times N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{B \times H \times N \times N}, \quad O = PV \in \mathbb{R}^{B \times H \times N \times d}
$$

- **总 FLOPs**: $FLOPs_{std} = 2 \cdot B \cdot H \cdot N^2 \cdot d \;(\text{GEMM } S) + 5 \cdot B \cdot H \cdot N^2 \;(\text{softmax}) + 2 \cdot B \cdot H \cdot N^2 \cdot d \;(\text{GEMM } O)$

  简化后主导项为：
  $$
  FLOPs_{std} \approx 4 \cdot B \cdot H \cdot N^2 \cdot d
  $$

- **总 HBM 访存**: 需读写 $Q, K, V, S, P, O$ 共 6 个张量。其中 $S, P$ 为 $N \times N$。

  $$
  Bytes_{std} \approx 2 \cdot B \cdot H \cdot N \cdot d \; (Q,K,V,O) + 4 \cdot B \cdot H \cdot N^2 \; (S,P)
  $$

  当 $N \gg d$ 时（如 $N=8192, d=64$），$Bytes_{std}$ 由 $O(N^2)$ 项主导。

- **算术强度**:
  $$
  AI_{std} = \frac{4 \cdot B \cdot H \cdot N^2 \cdot d}{4 \cdot B \cdot H \cdot N^2 + \text{低阶项}} \approx d \quad \text{当 } N \to \infty
  $$

  代入 $d=64$：
  $$
  AI_{std} \approx 64 \; \text{FLOP/Byte}
  $$

### 3.2 FlashAttention 的计算与访存

FlashAttention（以本实现 v0.4.0 为例，采用 online softmax + tiling，无中间矩阵 materialize）的核心不变量为：

- 将 $Q, K, V$ 分块为 SRAM 可容纳的 tile（如 $B_r \times d$, $B_c \times d$）。
- 仅输出 $O$ 写回 HBM；中间量 $S, P$ 在 SRAM 内生成、消费、丢弃。
- Online softmax 维护两个统计量：row max $m$ 与 row sum $l$。

**访存分析**（FA1 结构：一个 Q block 串行遍历全部 KV block，故 K/V 会按 Q block 数重载）：

| 数据 | 大小 | 方向 | 次数 | 说明 |
|------|------|------|------|------|
| $Q$ | $B \cdot H \cdot N \cdot d$ | HBM $\to$ SRAM | 1 | 逐 tile 读取 |
| $K$ | $B \cdot H \cdot N \cdot d$ | HBM $\to$ SRAM | $\lceil N / B_r \rceil$ | 每个 Q block 重读一遍 |
| $V$ | $B \cdot H \cdot N \cdot d$ | HBM $\to$ SRAM | $\lceil N / B_r \rceil$ | 与 $K$ 同步重载 |
| $O$ | $B \cdot H \cdot N \cdot d$ | SRAM $\to$ HBM | 1 | 最终输出 |
| $m, l$ | $B \cdot H \cdot N$ | SRAM（驻留） | 0 | 本实现在 tile 迭代中驻留 SRAM |

因此，总 HBM 流量（以元素计）为：

$$
\text{HBM 流量} = \underbrace{2 \cdot B \cdot H \cdot N \cdot d}_{Q,O} + \underbrace{2 \cdot B \cdot H \cdot N \cdot d \cdot \left\lceil \frac{N}{B_r} \right\rceil}_{K,V\;\text{重载}}
= \Theta\left(\frac{N^2 d}{B_r}\right)
$$

其中 $B_r = \text{BLOCK}_M$ 固定为 64/32，所以**本实现的 HBM 流量是 $O(N^2)$**，
只是常数小于 materialized attention 的 $O(N^2)$。

**causal 场景**：每个 Q block 只需读取其对角线之前的 KV 行，K/V 重载约减半：

$$
\text{HBM 流量}_{\text{causal}} \approx 2 \cdot B \cdot H \cdot N \cdot d + B \cdot H \cdot d \cdot \frac{N^2}{B_r}
$$

> **重要**：$\text{工作内存 / 中间激活}$ 是 $O(N)$（这是 FlashAttention 的核心收益），
> 但**不要**把"显存占用 $O(N)$"误写成"HBM 流量 $O(N)$"。本实现没有 split-KV，
> K/V 的重载次数不随 SRAM 容量增大而减少，因此 HBM 流量无法达到论文中
> $\Theta(N^2 d^2 / M)$ 的最优标度。

**算术强度**：

$$
AI_{FA} = \frac{FLOPs_{FA}}{\text{HBM Bytes}} \approx \frac{4 \cdot B \cdot H \cdot N^2 \cdot d}{c \cdot B \cdot H \cdot d \cdot \frac{N^2}{B_r} \cdot \text{bytes}} = \frac{4 B_r}{c \cdot \text{bytes}}
$$

其中 $c$ 为常数（$c \approx 2$，含 Q/O 项；$\text{bytes}$ 为每元素字节数）。以
$B_r = 64$、FP16（2 字节/元素）估算，$AI_{FA} \approx 4 \times 64 / (2 \times 2) = 64$
FLOP/Byte 量级——**与标准 attention 同阶**，远低于理论上的 $O(N)$。这正是固定 tile
（无 split-KV）与论文最优实现的本质差距。

> **注意**: 上述 $AI_{FA}$ 是**理论上限**，假设 K/V 完全复用（只在 HBM 读一次）。
> 实际 kernel 中，causal mask 的边界判断、softmax 的 online rescaling、以及 SRAM
> bank conflict 会导致有效 $AI$ 进一步下降。

### 3.3 为什么 FlashAttention 仍是 Memory-bound

本实现固定 tile（$B_r = 64/32$）下的理论 $AI_{FA}$ 与标准 attention 同阶（约几十
FLOP/Byte，见 3.2），在 Roofline 模型中必须区分**算法算术强度**与**有效算术强度**：

| 因素 | 对 $AI$ 的影响 | 说明 |
|------|--------------|------|
| Causal mask 不规则访存 | 降低 10%–20% | 下三角导致每个 query tile 需处理的 key tile 数量递减，warp 利用率不均 |
| Online softmax 额外 FLOPs | 提升 $AI$ | 重缩放、max 更新、log-sum-exp 增加少量计算，但不显著增加访存 |
| SRAM $\to$ Register / Shared Mem 流量 | **不纳入 HBM 流量** | Roofline 模型若使用 HBM-only 字节数，会高估 $AI$；若使用**全部内存层级流量**（含 shared memory），$AI$ 会大幅下降 |
| 小 head_dim（$d=32$） | 降低 $AI$ | 每个元素的计算量减少，tiling 粒度受限 |

**工程结论**: 本实现（无 split-KV，固定 tile）的 HBM 流量为 $O(N^2)$，算术强度不随
$N$ 增长，因此**深处于 memory-bound 区域**。只有 FlashAttention-2/3 通过 split-KV /
sequence-parallel 把 K/V 重载降下来，才能把 $AI$ 提高到 ridge point 附近。

> **面试核心论点**: FlashAttention 的优化目标不是"变成 compute-bound"，而是"在
> memory-bound 中做到最好"——工作内存从 $O(N^2)$ 降到 $O(N)$，使大 $N$ 变得可行；
> 但**要诚实地指出**：固定 tile 的实现 HBM 流量仍是 $O(N^2)$，进一步减少 K/V 重载
> 需要 split-KV（本仓库未实现）。

---

## 4. Tiling 如何提高算术强度并减少 HBM 流量

### 4.1 无 Tiling 的访存灾难

以 $N=16384, d=64, B=1, H=8$，FP16（2 字节/元素）为例。本实现的 HBM 流量按 3.2 的
K/V 重载模型计算（scalar 前向 hd64 取 $B_r = 64$，$q\_blocks = 256$）：

| 指标 | 标准 Attention | FlashAttention (tiled, 本实现) |
|:---|---:|---:|
| $S = QK^T$ 大小 | $8 \times 16384^2 \times 2 \text{ Bytes} = 4.29 \text{ GB}$ | 0（SRAM 内消纳） |
| $P = \text{softmax}(S)$ 大小 | $4.29 \text{ GB}$ | 0（SRAM 内消纳） |
| 总 HBM 激活内存 | ~8.6 GB（仅 $S, P$）+ 67 MB（$Q,K,V,O$） | ~67 MB（仅 $Q,K,V,O$ 与 $L$） |
| HBM 流量（读+写，单次前向）| ~17.2 GB | $Q(16.8\text{ MB}) + O(16.8\text{ MB}) + K(256 \times 16.8\text{ MB}) + V(256 \times 16.8\text{ MB}) \approx 8.6 \text{ GB}$ |
| 算术强度 $AI$ | $\approx d = 64$ | $\approx 64$（固定 tile，与标准同阶） |

> **关键更正**：本实现（FA1 结构，固定 tile，无 split-KV）的 HBM 流量**不是** 520 MB
> 量级，而是按 Q block 数重载 K/V 后的**数 GB 级**。FlashAttention 真正消除的是
> $O(N^2)$ 的**显存占用**（$S, P$ 不落 HBM），而不是 $O(N^2)$ 的 **HBM 流量**。
> causal 场景下 K/V 重载约减半，HBM 流量约为 4.3 GB。

### 4.2 Tiling 的算术强度提升机制

Tiling 提高 $AI$ 的本质是**数据复用（Data Reuse）**：

$$
AI = \frac{\text{FLOPs}}{\text{HBM Bytes}} = \frac{\text{FLOPs per tile} \times \text{num tiles}}{\text{HBM Bytes per tile} \times \text{num tiles}} \xrightarrow{\text{reuse}} \frac{\text{FLOPs per tile}}{\text{HBM Bytes per tile} / \text{reuse factor}}
$$

在 FlashAttention 中：

- 一个 $Q$ tile（$B_r \times d$）与所有 $K$ tiles 计算内积，产生 $B_r \times N$ 的局部 $S$ 行。
- 每个 $K$ tile（$B_c \times d$）被加载到 SRAM 后，服务于**多个** $Q$ tiles（若 non-causal）或**递减数量**的 $Q$ tiles（若 causal）。
- 计算量随 $B_r \times B_c \times d$ 增长，而 HBM 流量仅随 $B_r \times d + B_c \times d$ 增长。

**SRAM 容量约束**:

设 SRAM 大小为 $M_{SRAM}$（A100 每 SM 为 164 KB，可被多个 block 分区使用），则 tiling 需满足：

$$
\underbrace{B_r \times d}_{Q\;tile} + \underbrace{2 \times B_c \times d}_{K,V\;tiles} + \underbrace{B_r \times B_c}_{S\;tile} + \underbrace{B_r}_{m\;vector} + \underbrace{B_r}_{l\;vector} + \underbrace{B_r \times d}_{O\;accumulator} \leq M_{SRAM}
$$

本实现 v0.4.0 的 scalar 前向在 $d=64$ 时选取 $B_r = 64, B_c = 64$，则 SRAM 占用约为：

$$
64 \times 64 + 2 \times 64 \times 64 + 64 \times 64 + 64 + 64 + 64 \times 64 = 4\text{K} + 8\text{K} + 4\text{K} + 0.1\text{K} + 0.1\text{K} + 4\text{K} \approx 20\text{KB}
$$

（FP16/BF16 的 WMMA 前向 tile 更小：hd64 为 $64 \times 32$。）远小于默认 48 KB 动态
共享内存上限，留有余量给编译器插入的临时变量与 bank conflict 规避 padding。

---

## 5. 实测带宽利用率与 Roofline 定位

### 5.1 有效带宽利用率

以下数据是 v0.4.0 的历史快照（声称在 V100/A100/H100 上经 `nvprof`/`ncu` 采集）。
> **注意**：该快照的"理论 HBM 流量"列按旧的"Q/K/V/O 各读一次"模型估算，**已被
> 第 3.2 节的 K/V 重载模型取代**（固定 tile 下应为 $O(N^2)$ 量级），因此本表的
> 有效带宽/利用率数值只作为旧快照参考。本仓库实际硬件（RTX 3060 / sm_86）的
> 权威数据见 [validation-v1.0.md](./validation-v1.0.md)（T5 生成）。测试配置：
> batch=1, heads=8, head_dim=64, causal FP16。

| GPU | seq_len | 实测时间 (ms) | 理论 FLOPs | 实测 TFLOPS | 理论 HBM 流量 (GB) | 有效带宽 (GB/s) | 峰值带宽利用率 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| V100 | 1,024 | 0.42 | 2.15 | 2.1 | 0.23 | 548 | 61% |
| V100 | 4,096 | 5.82 | 34.4 | 2.8 | 0.92 | 630 | 70% |
| V100 | 8,192 | 22.50 | 137.4 | 3.0 | 1.84 | 651 | 72% |
| V100 | 16,384 | 88.0 | 549.8 | 3.1 | 3.68 | 670 | 74% |
| A100 | 1,024 | 0.19 | 2.15 | 4.5 | 0.23 | 1,211 | 59% |
| A100 | 4,096 | 2.18 | 34.4 | 7.5 | 0.92 | 1,631 | 80% |
| A100 | 8,192 | 7.80 | 137.4 | 8.5 | 1.84 | 1,855 | 91% |
| A100 | 16,384 | 28.5 | 549.8 | 9.3 | 3.68 | 1,957 | 96% |
| H100 | 1,024 | 0.11 | 2.15 | 8.2 | 0.23 | 2,091 | 62% |
| H100 | 4,096 | 1.15 | 34.4 | 14.2 | 0.92 | 3,020 | 90% |
| H100 | 8,192 | 3.85 | 137.4 | 17.3 | 1.84 | 3,247 | 97% |
| H100 | 16,384 | 13.2 | 549.8 | 20.1 | 3.68 | 3,350 | **100%** |

### 5.2 Roofline 图上定位

基于上表计算有效算术强度 $AI_{eff} = \text{实测 TFLOPS} \times 10^{12} / (\text{有效带宽} \times 10^9)$，并在 Roofline 坐标系中标定：

| GPU | seq_len | $AI_{eff}$ (FLOP/Byte) | Roofline Regime | 距离 Ridge Point |
|:---|:---:|:---:|:---|:---|
| V100 | 1,024 | 3.8 | Deep memory-bound | 9.2× 低于 ridge |
| V100 | 4,096 | 4.4 | Deep memory-bound | 7.9× 低于 ridge |
| V100 | 8,192 | 4.6 | Deep memory-bound | 7.6× 低于 ridge |
| V100 | 16,384 | 4.6 | Deep memory-bound | 7.6× 低于 ridge |
| A100 | 1,024 | 3.7 | Deep memory-bound | 41× 低于 ridge |
| A100 | 4,096 | 4.6 | Deep memory-bound | 33× 低于 ridge |
| A100 | 8,192 | 4.6 | Deep memory-bound | 33× 低于 ridge |
| A100 | 16,384 | 4.8 | Deep memory-bound | 32× 低于 ridge |
| H100 | 1,024 | 3.9 | Deep memory-bound | 76× 低于 ridge |
| H100 | 4,096 | 4.7 | Deep memory-bound | 63× 低于 ridge |
| H100 | 8,192 | 5.3 | Deep memory-bound | 56× 低于 ridge |
| H100 | 16,384 | 6.0 | Deep memory-bound | 49× 低于 ridge |

> **关键洞察**: $AI_{eff}$ 仅约 4–6 FLOP/Byte，远低于所有 GPU 的 ridge point。这意味着本实现 v0.4.0 的**有效**性能受限于带宽，但带宽利用率随 seq_len 增加而提高（因为固定开销被摊薄）。

### 5.3 为什么 $AI_{eff}$ 与理论 $AI_{FA}$ 差距巨大

第 3.2 节推导的本实现理论 $AI_{FA}$（固定 tile、FP16）约 64 FLOP/Byte 量级，而本仓库
实测的 $AI_{eff}$ 往往更低（memory-bound、且有效利用率低于理论）。差距原因如下：

| 因素 | 影响量级 | 解释 |
|------|:-------:|------|
| **HBM 流量定义差异** | $\times 2$–$10$ | 理论 $AI$ 只按逻辑字节（Q/K/V/O 元素数 × 字节数）计算；ncu 统计的物理 HBM 流量还包含 padding、非合并访问、kernel 启动参数等隐性流量。 |
| **Causal mask 不规则性** | $\times 1.5$–$2$ | Causal mask 导致大量 warp 内线程闲置（padding 至三角形边界），有效 FLOPs 降低。 |
| **Online softmax 额外访存** | $\times 1.2$ | $m, l$ 向量的频繁读写（即使驻留 SRAM，也有 register spilling 到 local memory 的情况）。 |
| **短序列固定开销** | $\times 2$–$4$ | `seq_len=1K` 时，kernel launch、grid setup、边界条件判断的 overhead 占比极高。 |

**正确的 HBM-only 算术强度口径**（本实现、固定 tile、无 split-KV）：

$$
AI_{HBM\text{-}only} = \frac{4 \cdot B \cdot H \cdot N^2 \cdot d}{2 \cdot B \cdot H \cdot N \cdot d \; (Q,O) + 2 \cdot B \cdot H \cdot N \cdot d \cdot \lceil N / B_r \rceil \; (K,V\;\text{重载})}
\approx \frac{2 B_r}{\text{bytes}}
$$

与标准 attention 的 $AI \approx d$ 同阶（$B_r=64$、FP16 时约 64 FLOP/Byte），**不随
$N$ 增长**。这与"$AI \approx O(N)$"的旧结论完全不同——旧结论假设 K/V 只读一次，
对本实现（无 split-KV）不成立。要让 $AI$ 随 $N$ 增长，必须引入 split-KV /
sequence-parallel（FlashAttention-2/3），把 K/V 重载降下来。

---

## 6. 标准 Attention vs FlashAttention 的 Roofline 对比

### 6.1 同一坐标系下的定位

以 A100（$\beta_{peak}=2039$ GB/s, $\pi_{peak}=312$ TFLOPS, $AI_{ridge}=153$）为基准：

```
Performance (TFLOPS)
    |
312 |______________________________  Compute Roof (Flat)
    |                             /
    |                           /
    |                         /
    |                       /   <-- Ridge Point @ AI=153
    |                     /
    |                   /
    |                 /
    |               /
    |             /  <-- Bandwidth Roof (Slope = 2039 GB/s)
    |           /
    |         /
    |       /
    |     /
    |   /
    | /
    +-----------------------------------> AI (FLOP/Byte)
      1    10    50   100   153   500   1000

Standard Attention (seq=16K):  X @ AI≈64,  P≈0.13 TFLOPS
FlashAttention (本实现, seq=16K): O @ AI≈64 (固定 tile, O(N²) HBM), P≈9.3 TFLOPS (v0.4.0 快照)
FlashAttention-2 (参考):        △ @ AI≈O(N),  P≈80+ TFLOPS

* 本实现 HBM 流量为 O(N²)，AI 不随 N 增长；FA2/3 的 split-KV 才降到 O(N)。
```

### 6.2 对比汇总表

| 维度 | 标准 Attention (Materialized) | CuFlash-Attn（本实现，固定 tile） | FlashAttention-2/3 (生产级) |
|:---|:---|:---|:---|
| $AI$ (HBM-only) | $O(d) \approx 64$ | $O(2B_r/\text{bytes}) \approx 64$（固定 tile，**不随 $N$ 增长**） | $O(N/B_c)$（split-KV 后大幅提高） |
| HBM 流量 scaling | $O(N^2)$ | **$O(N^2)$**（固定 tile，常数更小） | **$O(N)$** / split-KV 后大幅降低 |
| 工作内存（显存占用） | $O(N^2)$ | $O(N)$ | $O(N)$ |
| A100 峰值带宽利用率 * | 20%–35% | 60%–96%（v0.4.0 历史快照） | 85%–110% |
| A100 实测 TFLOPS * | 1.5–3.0 | 4.5–9.3（v0.4.0 历史快照） | 80–150+ |
| 最大 seq_len (40GB) | ~8K–16K | ~64K | ~128K–256K |
| Roofline Regime | Deep memory-bound, 低效 | Memory-bound | Near ridge point / 部分 compute-bound |

> * 带 * 的行是 v0.4.0 历史快照（非本仓库硬件实测），仅作量级参考；权威实测见
> [validation-v1.0.md](./validation-v1.0.md)。

### 6.3 定性结论

1. **标准 Attention** 位于 Roofline 极左下角。即使给它无限算力，也无法突破 $P = \beta \times AI$ 的斜线限制；且 $AI$ 固定为 $O(d)$，不随 $N$ 增长，**不具备 scaling 潜力**。

2. **本实现（固定 tile）** 的核心收益是**工作内存**从 $O(N^2)$ 降到 $O(N)$，使长序列
   变得可行；但 HBM 流量仍是 $O(N^2)$，$AI$ 与标准 attention 同阶、不随 $N$ 增长，
   因此深处于 memory-bound 区域。这是 FA1 结构 + 固定 tile 的固有上限。

3. **FlashAttention-2/3** 通过以下手段真正把 HBM 流量降下来：
   - **Split-K / Sequence Parallel**: 将 $K, V$ 的冗余加载分摊到多个 warp group，把
     HBM 流量从 $O(N^2)$ 降到 $O(N)$。
   - **Grouped GEMM / Warp Specialization**: 减少 softmax 与 GEMM 之间的流水线气泡。
   - **TMA (Hopper) / cp.async (Ampere)**: 异步预取隐藏 HBM 延迟。
   - **精确 causal mask 处理**: 避免 tile 内的无效计算与访存。

   这些优化使得生产级 FlashAttention 在 A100 上可达到 ridge point 附近，在 H100 上配合 TMA 甚至部分进入 compute-bound regime。

---

## 7. 优化路线图（从 Roofline 视角）

> 本仓库 v1.0 冻结为教学/参考实现，以下路线图**不在本仓库实施**（见 PLAN.md 的
> Phase B 界定），仅记录从 Roofline 视角看后续可以怎么走。

| 阶段 | 目标 | 手段 | 预期收益 | 难度 |
|:---|:---|:---|:---:|:---:|
| v1.0 (当前) | 正确、可复现的基线 | 基础 tiling + online softmax + 前向 WMMA | 工作内存 $O(N)$，HBM 流量 $O(N^2)$（诚实口径） | 基线 |
| 优化 1 | 降低 K/V 重载 | `cp.async` 预取、更优 warp 调度、causal mask 边界优化 | 隐藏 HBM 延迟 | 中 |
| 优化 2 | 把 HBM 流量降到 $O(N)$ | split-KV / sequence-parallel、warp-group 级 reduction | $AI$ 随 $N$ 增长，接近 ridge point | 高 |
| 未来 | 接近 ridge point | CUTLASS 集成或 TMA/WGMMA 重写（Hopper） | 生产级 | 极高 |

---

## 8. 参考公式速查

| 符号 | 定义 | 单位 |
|:---|:---|:---|
| $N$ | 序列长度（`seq_len`） | — |
| $d$ | 头维度（`head_dim`） | — |
| $B$ | batch size | — |
| $H$ | 注意力头数 | — |
| $B_r, B_c$ | Query / Key-Value tile 大小 | — |
| $\beta_{peak}$ | HBM 峰值带宽 | GB/s |
| $\pi_{peak}$ | FP16 Tensor Core 峰值算力 | TFLOPS |
| $AI$ | 算术强度 = FLOPs / Bytes | FLOP/Byte |
| $AI_{ridge}$ | Ridge point = $\pi_{peak} / \beta_{peak}$ | FLOP/Byte |
| $P_{roofline}$ | Roofline 性能上限 = $\min(\pi_{peak}, \beta_{peak} \times AI)$ | TFLOPS |

---

## 9. 推荐阅读

1. Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An insightful visual performance model for multicore architectures*. Communications of the ACM.
2. Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.
3. Dao, T., et al. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.
4. NVIDIA. (2022). *CUDA C++ Programming Guide* — Compute Capability 8.0/9.0 Architecture Details.
