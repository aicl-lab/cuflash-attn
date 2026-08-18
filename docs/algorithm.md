# FlashAttention 算法详解

FlashAttention 是一种 IO-aware 的精确注意力算法，将内存复杂度从 $O(N^2)$ 降至 $O(N)$，同时数值上严格等价于标准注意力。

---

## 目录

- [标准注意力瓶颈](#标准注意力瓶颈)
- [核心 FlashAttention 概念](#核心-flashattention-概念)
  - [分块 (Tiling)](#1-分块-tiling)
  - [Online Softmax](#2-online-softmax)
  - [重计算 (Recomputation)](#3-重计算-recomputation)
- [前向传播算法](#前向传播算法)
- [反向传播算法](#反向传播算法)
- [因果掩码](#因果掩码)
- [FP16/BF16 实现](#fp16bf16-实现)
- [内存复杂度分析](#内存复杂度分析)
- [实现亮点](#实现亮点)
- [参考文献](#参考文献)

---

## 标准注意力瓶颈

标准自注意力定义为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

这会产生三个需要物化的中间矩阵：

$$
S = QK^T \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = PV \in \mathbb{R}^{N \times d}
$$

**核心问题：** $S$ 和 $P$ 具有 $O(N^2)$ 大小，必须存放在 HBM（设备内存）中。对于大的 $N$：

| 问题 | 影响 |
|------|------|
| **内存占用** | $N=4096$, 32 heads $\Rightarrow$ 仅 $S$ 和 $P$ 就约 2 GB |
| **带宽瓶颈** | GPU 算力 $\gg$ HBM 带宽；时间由数据搬运主导 |
| **IO 操作** | $S$ 和 $P$ 各需写入和读出 HBM：共 4 次 $O(N^2)$ 操作 |

![分块概览](/diagrams/tiling-overview.svg)

*图 1：Q/K/V 分块加载到 SRAM。中间量 $S$ 和 $P$ 永不触碰 HBM。*

---

## 核心 FlashAttention 概念

### 1. 分块 (Tiling)

将 $Q$、$K$、$V$ 切分为可放入 SRAM（共享内存 / L1 缓存）的块：

$$
Q = [Q_1, Q_2, \ldots, Q_{T_r}], \quad Q_i \in \mathbb{R}^{B_r \times d}
$$

$$
K = [K_1, K_2, \ldots, K_{T_c}], \quad K_j \in \mathbb{R}^{B_c \times d}
$$

$$
V = [V_1, V_2, \ldots, V_{T_c}], \quad V_j \in \mathbb{R}^{B_c \times d}
$$

**分块大小选择：**

| GPU 架构 | SRAM 容量 | 典型 $B_r \times B_c$ |
|----------|-----------|----------------------|
| Volta (V100) | 96 KB | $64 \times 64$ |
| Ampere (A100) | 164 KB | $128 \times 64$ |
| Hopper (H100) | 228 KB | $128 \times 128$ |

**分块为何有效：**
- 每块放入高速 SRAM（$\sim$19 TB/s）而非低速 HBM（$\sim$2 TB/s）。
- 避免中间结果反复访问 HBM。
- 独立的 Q 块可并行处理。

### 2. Online Softmax

标准 softmax 需要对每行两次遍历（找 $\max$ $\to$ 计算 $\exp$ 和 $\to$ 归一化）。FlashAttention 使用 **online softmax**，在单次遍历 KV 块的过程中增量更新：

对每个 KV 块 $j$，由 Q 块 $i$ 处理：

$$
m_{ij}^{\text{new}} = \max(m_{ij}^{\text{old}}, \text{rowmax}(S_{ij}))
$$

$$
\tilde{P}_{ij} = \exp(S_{ij} - m_{ij}^{\text{new}})
$$

$$
l_{ij}^{\text{new}} = \exp(m_{ij}^{\text{old}} - m_{ij}^{\text{new}}) \cdot l_{ij}^{\text{old}} + \text{rowsum}(\tilde{P}_{ij})
$$

$$
O_{ij}^{\text{new}} = \frac{\exp(m_{ij}^{\text{old}} - m_{ij}^{\text{new}}) \cdot O_{ij}^{\text{old}} + \tilde{P}_{ij} V_j}{l_{ij}^{\text{new}}}
$$

![Online Softmax 状态机](/diagrams/online-softmax-state-machine.svg)

*图 2：Online softmax 状态更新。当新的 KV 块揭示更大的行最大值时，先前输出被 $\exp(m_{\text{old}} - m_{\text{new}})$ 重新缩放。*

**关键洞察：** 处理新 KV 块时，若全局行最大值改变，先前输出必须通过 $\exp(m_{\text{old}} - m_{\text{new}})$ 修正。

**数值稳定性：** 追踪运行最大值确保 $\exp(\cdot)$ 永不过溢，即使注意力分数很大。

### 3. 重计算 (Recomputation)

标准反向传播存储 $O(N^2)$ 的注意力矩阵 $P$ 用于梯度计算。FlashAttention 的策略：

| 阶段 | 存储内容 | 内存 |
|------|----------|------|
| **前向** | 仅输出 $O$ 和 logsumexp $L$ | $O(N)$ |
| **反向** | 从 $Q, K, V, O, L$ 实时重计算 $P$ | $O(N)$ |

**权衡：** 增加约 33% 的额外 FLOPs，但显著减少 HBM IO，整体仍加速。

![反向重计算数据流](/diagrams/backward-recompute-flow.svg)

*图 3：反向传播在 SRAM 中从 forward 输出重计算 $P_{ij}$。不存储 $O(N^2)$ 矩阵。*

---

## 前向传播算法

```
输入:  Q, K, V ∈ R^(N×d), scale = 1/√d
输出: O ∈ R^(N×d), L ∈ R^N

初始化: O = 0, m = -∞, l = 0  (每行)

对每个 Q 块 i (i = 1..T_r 并行):
    将 Q_i 加载到 SRAM
    对每个 KV 块 j = 1..T_c (顺序):
        将 K_j, V_j 加载到 SRAM
        S_ij = scale × Q_i × K_j^T           # [B_r, B_c] 在 SRAM 中
        m_new = max(m_i, rowmax(S_ij))       # 更新行最大值
        P = exp(S_ij - m_new)                 # 局部 softmax 分子
        l_new = exp(m_i - m_new) × l_i + rowsum(P)
        O_i = (exp(m_i - m_new) × O_i + P × V_j) / l_new
        m_i = m_new, l_i = l_new
    L_i = m_i + log(l_i)                      # 存储 logsumexp
```

**关键操作：**
1. **Q 块并行：** 每个输出块由一个 CUDA block 独立计算。
2. **KV 块顺序：** 跨所有 key 累积注意力。
3. **输出修正：** 发现新最大值时调整运行和。

---

## 反向传播算法

```
输入:  Q, K, V, O, L, dO
输出: dQ, dK, dV

对每个 KV 块 j:
    将 K_j, V_j 加载到 SRAM
    初始化 dK_j = 0, dV_j = 0
    对每个 Q 块 i:
        将 Q_i, O_i, dO_i, L_i 加载到 SRAM
        S_ij = scale × Q_i × K_j^T
        P_ij = exp(S_ij - L_i)                # 重计算注意力权重
        D_i = rowsum(dO_i ⊙ O_i)              # 对角项
        dV_j += P_ij^T × dO_i                 # V 梯度
        dP_ij = dO_i × V_j^T
        dS_ij = P_ij ⊙ (dP_ij - D_i)          # Softmax Jacobian
        dQ_i += scale × dS_ij × K_j           # Q 梯度
        dK_j += scale × dS_ij^T × Q_i        # K 梯度
```

**梯度流：**
1. **dV：** 使用重计算的注意力权重对上游梯度加权求和。
2. **dQ, dK：** 通过 softmax Jacobian，使用重计算的 $P$。
3. **内存高效：** 任何时刻都不需要 $O(N^2)$ 存储。

---

## 因果掩码

对于自回归模型，位置 $i$ 只能 attend 到位置 $\leq i$。FlashAttention 的块结构支持高效因果掩码：

| 情况 | 处理方式 |
|------|----------|
| **完全跳过** | KV 块起始列 $>$ Q 块结束行 $\Rightarrow$ 跳过整个块 |
| **部分掩码** | 块内应用掩码（设为 $-\infty$） |

**效率提升：** 约 50% 的块可完全跳过，计算量减少一半。

![因果掩码块](/diagrams/causal-masking-blocks.svg)

*图 4：块级因果掩码。下三角块完全计算；对角块部分掩码；上三角块跳过。*

---

## FP16/BF16 实现

本实现支持 FP32 / FP16 / BF16 的前向与反向传播。自 v0.5.0 起，FP16（sm_70+）与
BF16（sm_80+）的**前向**在支持 Tensor Core 的架构上走 **WMMA 路径**。

### WMMA 前向（默认路径，sm_70+ / sm_80+）

- Q/K/V tile 在 shared memory 中**保持输入精度**（`half` / `__nv_bfloat16`）；
- $S = QK^T$ 与 $O \mathrel{+}= P V$ 的矩阵乘法在 Tensor Core 上执行
  （m16n16k16，FP32 累加）；
- online softmax 与 FA2 延迟归一化仍在 CUDA core 上以 FP32 完成；
- softmax 后把 $P$ 量化为**输入精度**再做 $PV$。

$$
\text{输入: } \texttt{half}^*/\texttt{bf16}^*\; Q, K, V
\xrightarrow{\text{加载}} \text{输入精度 tile} \xrightarrow{\text{MMA (FP32 累加)}} \text{FP32 累加器}
\xrightarrow{\text{存储}} \texttt{half}^*/\texttt{bf16}^*\; O,\; \texttt{float}^*\; L
$$

### Scalar fallback（FP32 全程；或低精度在不支持 WMMA 的架构上）

- 输入在加载时转为 FP32，CUDA core 计算，FP32 累加，输出转回输入精度：

$$
\text{输入: } \texttt{half}^* \; Q, K, V \xrightarrow{\text{加载}} \text{FP32 寄存器} \xrightarrow{\text{计算}} \text{FP32 累加器} \xrightarrow{\text{存储}} \texttt{half}^* \; O, L
$$

**反向**（三 dtype 一致）目前全部为 scalar 路径：输入加载转 FP32、FP32 累加。

### 数值精度

| 操作 | WMMA 前向 | scalar 路径 |
|------|----------|-------------|
| 矩阵乘法 ($Q \times K^T$) | Tensor Core，FP32 累加 | FP32 |
| Softmax 计算 | FP32 | FP32 |
| 累加 | FP32 | FP32 |
| 最终输出 | FP16/BF16 | FP16/BF16 |

**优势：**
- 低精度输入把 HBM 流量减半（张量缩小 2 倍）。
- WMMA 路径把两次大矩阵乘放到 Tensor Core 上（FP16 需 compute capability $\geq$ 7.0，BF16 需 $\geq$ 8.0）。

---

## 内存复杂度分析

> 必须区分两种不同的量：
> - **工作内存 / 中间激活**（SRAM 之外需要物化的数据量）是 $O(N)$；
> - **HBM 流量**（实际读写设备内存的字节数）对本实现**仍是 $O(N^2)$**，
>   只是常数小于 materialized attention。

| 方法 | 前向内存 | 反向内存 | HBM 流量 |
|------|----------|----------|--------|
| 标准注意力 | $O(N^2)$ | $O(N^2)$ | $O(N^2 + Nd)$ |
| 本实现（固定 tile） | $O(N)$ | $O(N)$ | $\Theta\left(\frac{N^2 D}{\text{BLOCK}_M} + ND\right)$（固定 tile 下即 $O(N^2)$） |

本实现是 FA1 结构：一个 Q block 串行遍历全部 KV block，Q 读 1 次、O 写 1 次，
每个 Q block 都会把 K/V 各重读一遍：

$$
\text{HBM 流量} = Q(1\text{ 次}) + O(1\text{ 次}) + (K+V) \times \left\lceil \frac{N}{\text{BLOCK}_M} \right\rceil
= \Theta\left(\frac{N^2 D}{\text{BLOCK}_M}\right)
$$

其中 $\text{BLOCK}_M$ 固定为 64/32。论文中 $\Theta(N^2 D^2 / M)$ 的结论需要 block
size 随 SRAM 容量合理放大（即 $M$ 与 block 大小同阶增长）才成立；当前固定 tile
的实现并没有达到。FlashAttention-2 的 split-KV 进一步减少 K/V 重载，本仓库**没有实现**。

### 实际内存节省

| 序列长度 | 标准注意力 | FlashAttention | 节省 |
|----------|-----------|---------------|------|
| 1,024 | 4 MB | 8 KB | **99.8%** |
| 4,096 | 64 MB | 32 KB | **99.95%** |
| 16,384 | 1 GB | 128 KB | **99.99%** |

> 上表是**显存占用**的对比，不是 HBM 流量。工作内存为 $O(N)$ 是 FlashAttention
> 的核心收益；但本实现（无 split-KV）的 HBM 流量仍为 $O(N^2)$，见上文。

---

## 实现亮点

### 分块配置

前向 tile 由 `impl::ForwardTilingConfig` 集中定义（scalar 与 WMMA 两套）：

| head_dim | scalar 前向 $B_r \times B_c$ | WMMA 前向 $B_r \times B_c$ |
|----------|-----------------------------|----------------------------|
| 32 | 64 × 64 | 64 × 32 |
| 64 | 64 × 64（sm_75 等回退 32 × 32） | 64 × 32 |
| 128 | 32 × 32（回退 16 × 16） | 32 × 32 |

反向使用更保守的 `BackwardTilingConfig`（如 hd128 为 16 × 32），因为 shared
memory 需要容纳更多梯度张量。共享内存按 head_dim 运行时动态分配。

### 优化技术

| 技术 | 收益 |
|------|------|
| **向量化内存访问** | `float4` 加载/存储实现合并访问 |
| **Launch Bounds** | `__launch_bounds__(128)` 控制寄存器压力 |
| **动态共享内存** | 根据 head_dim 运行时分配 |
| **流安全** | 显式 workspace 生命周期管理 |
| **Warp 级原语** | `__shfl_sync` 用于 warp 内归约 |

### 数据类型支持

| 数据类型 | 前向 | 反向 |
|----------|------|------|
| FP32 (`float`) | 完整（scalar） | 完整（scalar） |
| FP16 (`half`) | 完整（WMMA / scalar fallback） | 完整（scalar） |
| BF16 (`__nv_bfloat16`) | 完整（WMMA / scalar fallback） | 完整（scalar） |

---

## 参考文献

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
   - Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
   - NeurIPS 2022
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   - Tri Dao
   - ICLR 2024
   - [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

3. **Online normalizer calculation for softmax**
   - Maxim Milakov, Natalia Gimelshein
   - [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

4. **NVIDIA CUDA Programming Guide - Shared Memory**
   - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
