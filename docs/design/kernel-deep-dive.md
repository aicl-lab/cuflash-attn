# Kernel 逐行解读

> **适用范围说明**：本文档逐行解读的是 **scalar（CUDA core）路径**——FP32 前向/反向
> 全部，以及低精度在不支持 WMMA 的架构上的 fallback。自 v0.5.0 起，FP16/BF16 **前向**
> 在 sm_70+/sm_80+ 上默认走 **WMMA（Tensor Core）路径**
> （`src/forward/flash_attention_forward_wmma.cu`）：Q/K/V tile 保持输入精度、
> 两个大矩阵乘在 Tensor Core 上执行（FP32 累加），online softmax 与延迟归一化仍在
> CUDA core 上完成。本文档下述的 SMEM 布局（全 `float`）与 `matmul_ABt` 仅对应 scalar
> 路径；WMMA 路径的布局与分派见 [tensor-core-migration.md](./tensor-core-migration.md)。

本文档逐行拆解 CuFlash-Attn 的 CUDA kernel 实现，覆盖从 launch configuration 到 warp-level 指令流水的全部细节。所有结论均直接对应源码，不作模糊表述。

---

## 目录

- [Kernel Launch 配置](#kernel-launch-配置)
- [Shared Memory 布局图解](#shared-memory-布局图解)
- [Warp-level 分工](#warp-level-分工)
- [Launch Bounds 与寄存器压力](#launch-bounds-与寄存器压力)
- [向量化内存访问与 Coalescing](#向量化内存访问与-coalescing)
- [Causal Masking 的 Warp-level 跳过逻辑](#causal-masking-的-warp-level-跳过逻辑)
- [FP16 内部 FP32 Accumulation 路径](#fp16-内部-fp32-accumulation-路径)
- [参考文献](#参考文献)

---

## Kernel Launch 配置

### Grid / Block 设定

```cpp
dim3 grid(num_q_blocks, batch_heads);
dim3 block(128);
```

| 参数 | 值 | 语义 |
|------|-----|------|
| `grid.x` | `ceil(seq_len / BLOCK_M)` | Q 方向分块数 |
| `grid.y` | `batch_size * num_heads` | 每个 batch×head 独占一个 CUDA block |
| `block.x` | `128` | 每 block 固定 128 线程 |
| `__launch_bounds__` | `128` | 每 block 最大线程数约束，编译器据此分配寄存器 |

**调度特性：**

- `grid.y` 维度的并行度天然与 batch×heads 成正比，无需额外拆分。
- 每个 block 处理一个 `(BLOCK_M, HEAD_DIM)` 的 Q tile，沿 KV 方向串行迭代。
- 当 `head_dim == 128` 时，BLOCK_M / BLOCK_N 从 64 下调至 32，以控制共享内存占用。

```cpp
// head_dim == 32 / 64
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;

// head_dim == 128
constexpr int BLOCK_M_HD128 = 32;
constexpr int BLOCK_N_HD128 = 32;
```

### Kernel 签名

```cpp
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
flash_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int seq_len, float scale, bool causal
);
```

模板三参数 `BLOCK_M`、`BLOCK_N`、`HEAD_DIM` 在编译期实例化，确保所有 tile 尺寸、循环边界、共享内存偏移均为常量，便于编译器展开和寄存器分配。

---

## Shared Memory 布局图解

### 前向传播布局

每个 block 通过 `extern __shared__ float smem[]` 动态申请共享内存，逻辑分区如下（以 `head_dim=64, BLOCK_M=BLOCK_N=64` 为例）：

```cpp
extern __shared__ float smem[];
float* Q_tile = smem;                           // [BLOCK_M, HEAD_DIM]
float* K_tile = Q_tile + BLOCK_M * HEAD_DIM;    // [BLOCK_N, HEAD_DIM]
float* V_tile = K_tile + BLOCK_N * HEAD_DIM;    // [BLOCK_N, HEAD_DIM]
float* S_tile = V_tile + BLOCK_N * HEAD_DIM;    // [BLOCK_M, BLOCK_N]
float* O_tile = S_tile + BLOCK_M * BLOCK_N;     // [BLOCK_M, HEAD_DIM]
float* m_tile = O_tile + BLOCK_M * HEAD_DIM;    // [BLOCK_M]
float* l_tile = m_tile + BLOCK_M;               // [BLOCK_M]
```

### 内存占用计算

| head_dim | BLOCK_M | BLOCK_N | 共享内存总量 (float) | 字节数 |
|----------|---------|---------|----------------------|--------|
| 32 | 64 | 64 | 64×32 + 64×32 + 64×32 + 64×64 + 64×32 + 64 + 64 = 12,928 | ~50.5 KB |
| 64 | 64 | 64 | 64×64 + 64×64 + 64×64 + 64×64 + 64×64 + 64 + 64 = 20,736 | ~81.0 KB |
| 128 | 32 | 32 | 32×128 + 32×128 + 32×128 + 32×32 + 32×128 + 32 + 32 = 13,376 | ~52.3 KB |

**关键观察：**

- `head_dim=64` 时共享内存最大（约 81 KB），仍低于 A100 每 block 164 KB 上限，但超过默认 48 KB 动态共享内存阈值，需通过 `cudaFuncSetAttribute` 申请 opt-in。
- `head_dim=128` 时通过缩小 BLOCK_M/BLOCK_N 至 32，将共享内存压回约 52 KB，换取对更大 head_dim 的支持。

### 布局示意图

```
低地址 ──────────────────────────────────────────────> 高地址
┌─────────┬─────────┬─────────┬─────────┬─────────┬──────┬──────┐
│  Q_tile │  K_tile │  V_tile │  S_tile │  O_tile │m_tile│l_tile│
│[64, 64] │[64, 64] │[64, 64] │[64, 64] │[64, 64] │ [64] │ [64] │
└─────────┴─────────┴─────────┴─────────┴─────────┴──────┴──────┘
  ↑       ↑         ↑         ↑         ↑         ↑      ↑
 smem   Q+...    Q+2×...   Q+3×...   Q+4×...   Q+5×...  +6×...
```

所有 tile 均为行主序（row-major）连续存储。`S_tile` 位于中间区域，在反向传播中被复用为 `dS_tile`。

---

## Warp-level 分工

### Warp 组织

128 线程 = 4 个 warp（每个 warp 32 线程），无显式 warp-specialization，全部线程通过循环步长协作：

```cpp
const int tid = threadIdx.x;
const int num_threads = blockDim.x;  // 128

for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
    // 每个线程处理全局索引 i, i+128, i+256, ...
}
```

### 各阶段分工表

| 阶段 | 操作 | 线程分配策略 | 同步点 |
|------|------|-------------|--------|
| **Load Q** | GMEM → SMEM | 128 线程按 `tid` 步进扫过 `BLOCK_M × HEAD_DIM` | `__syncthreads()` |
| **Load K/V** | GMEM → SMEM | 同上，扫过 `BLOCK_N × HEAD_DIM` | `__syncthreads()` |
| **Matmul Q@K^T** | SMEM → SMEM | 每个线程负责 `S_tile` 中一个或多个 `(row, col)` | `__syncthreads()` |
| **Causal Mask** | SMEM 内改写 | 128 线程扫过 `BLOCK_M × BLOCK_N`，条件写 `-INFINITY` | `__syncthreads()` |
| **Row-wise Max/Sum** | SMEM 内归约 | 每行由一个线程主导遍历 `BLOCK_N` 列 | 无（行间独立） |
| **Online Softmax + O 更新** | SMEM → SMEM | 每行一个线程，内循环 `HEAD_DIM` | `__syncthreads()` |
| **Store O** | SMEM → GMEM | 128 线程步进扫过 `BLOCK_M × HEAD_DIM` | 无（kernel 末尾） |

### Warp 利用率分析

- **Load/Store 阶段**：128 线程同时发起全局内存请求，理想情况下若满足对齐与连续条件，可合并为 128/4 = 32 个 `float4` 事务，占满 L2 带宽。
- **Compute 阶段**：`matmul_ABt` 中每个线程独立完成 dot-product，无 warp 内通信，因此 `__shfl` 系列原语在前向 kernel 中未被直接调用（但 `online_softmax.cuh` 中提供了 `warp_reduce_max` / `warp_reduce_sum` 供其他用途）。
- **分支发散**：Causal mask 阶段存在条件分支 `if (global_k > global_q)`，但由于被掩码元素统一写 `-INFINITY`，发散程度低；block-level 跳过逻辑（见后）进一步减少了进入此分支的 block 数。

---

## Launch Bounds 与寄存器压力

### `__launch_bounds__(128)` 的作用

```cpp
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
flash_attention_forward_kernel(...)
```

`__launch_bounds__(max_threads)` 向编译器承诺：每个 block 的活跃线程数不超过 128。编译器据此进行寄存器分配，核心公式为：

$$
\text{max_registers_per_thread} = \frac{\text{total_registers_per_SM}}{\text{max_blocks_per_SM} \times 128}
$$

### 寄存器压力对比

| 配置 | 每线程寄存器上限（理论） | occupancy 影响 |
|------|------------------------|---------------|
| 无 `__launch_bounds__` | 编译器保守估算，可能保留更多寄存器 | 单 SM 驻留 block 数受限 |
| `__launch_bounds__(128)` | 编译器已知最大并发线程密度，可收紧寄存器预算 | 提升 SM 占用率，允许更多 block 并行 |

**实际效果：**

- 限制寄存器使用量，降低单个 thread 的上下文体积，使得在共享内存允许的前提下，更多 block 能够同时驻留于同一 SM。
- 对于本实现，每个 thread 的 live variable 集较小（主要包含若干 float 累加器与索引变量），`__launch_bounds__(128)` 足以将寄存器数量控制在编译器可合理分配的范围内，无需进一步缩小至 64 或更小。

---

## 向量化内存访问与 Coalescing

### `float4` 向量化加载

```cpp
const bool can_vectorize = (BLOCK_COLS % 4 == 0) && (src_stride % 4 == 0) &&
                           (col_start % 4 == 0) && is_aligned_16(src) && is_aligned_16(dst);

if (can_vectorize) {
    const int total_vec = total_elements / 4;
    for (int i = tid; i < total_vec; i += num_threads) {
        int elem_idx = i * 4;
        int local_row = elem_idx / BLOCK_COLS;
        int local_col = elem_idx % BLOCK_COLS;
        int global_row = row_start + local_row;
        int global_col = col_start + local_col;

        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (global_row < max_rows && global_col + 3 < max_cols) {
            val = *reinterpret_cast<const float4*>(
                &src[global_row * src_stride + global_col]);
        }
        *reinterpret_cast<float4*>(&dst[local_row * BLOCK_COLS + local_col]) = val;
    }
}
```

### Coalescing 分析

**理想情形（向量化路径）：**

- `head_dim ∈ {32, 64, 128}` 均能被 4 整除。
- 张量布局为 `[batch, heads, seq_len, head_dim]` 连续存储，`src_stride = head_dim`，同样可被 4 整除。
- 若 `col_start == 0` 且全局指针 16 字节对齐（`cudaMalloc` 默认保证），则相邻线程（`tid` 与 `tid+1`）访问的 `float4` 地址连续。
- 128 线程同时发起访问时，Warp 0（tid 0–31）访问地址 `[0, 16), [16, 32), ..., [496, 512)`，合并为最少事务数。

**边界退化情形：**

- 当 `seq_len` 非 `BLOCK_M` / `BLOCK_N` 整数倍时，最后一个 tile 存在越界行/列。
- 越界行触发 `else if (global_row < max_rows)` 分支，退化为标量填充，仅影响该 warp 的尾部线程。
- 对于 `global_col + 3 >= max_cols` 的边界列，同样退化为逐元素复制，但这类元素占比随 `seq_len` 增大而迅速降低。

### 带宽利用率表

| 路径 | 每个线程事务数 / 4 float | 128 线程合并后最小事务数 | 适用条件 |
|------|------------------------|------------------------|---------|
| `float4` 全命中 | 1 | 32 (4 warps × 8 事务) | 行内列完全在界内 |
| `float4` 部分退化 | 1–4 标量 | 32–128 | 边界 tile，部分列越界 |
| 标量回退 | 4 | 128 | 未满足对齐条件（极少发生） |

---

## Causal Masking 的 Warp-level 跳过逻辑

### 两级跳过策略

```cpp
// Level 1: Block-level 跳过
if (causal && kv_start > q_start + BLOCK_M - 1) {
    break;  // 整个 KV block 位于未来，直接终止循环
}

// Level 2: Element-level 掩码
if (causal) {
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
        int q_idx = i / BLOCK_N;
        int k_idx = i % BLOCK_N;
        int global_q = q_start + q_idx;
        int global_k = kv_start + k_idx;
        if (global_k > global_q) {
            S_tile[i] = -INFINITY;
        }
    }
    __syncthreads();
}
```

### 计算量削减分析

设 `seq_len = N`，`BLOCK_M = BLOCK_N = B`。标准 Attention 的因果 mask 需处理 `N(N+1)/2` 个有效元素，本实现的两级策略进一步减少实际运算：

| 阶段 | 行为 | 跳过比例（近似） |
|------|------|----------------|
| Block-level | 当 `kv_block` 起始列 > `q_block` 结束行时，整 block `break` | 约 50% KV blocks 被完全跳过 |
| Element-level | 仅对“块内上三角”区域写 `-INFINITY` | 每 block 内约 `B²/2` 元素被掩码 |

**Warp 级效率：**

- Block-level `break` 使得 warp 直接退出 KV 循环，无需执行任何加载/计算/存储。
- Element-level mask 中，被掩码元素仍参与内存读写，但通过统一写 `-INFINITY` 保证 warp 内分支一致性高。未被掩码元素的比例沿对角线块递增。

### 因果 mask 的数学等价性

设原始 attention 分数为 `S_ij = Q_i · K_j^T * scale`。 causal masking 将 `j > i` 的位置设为 `-INFINITY`，softmax 后对应概率为 0。在分块实现中：

1. Block-level 跳过等价于识别出整块 `(i_block, j_block)` 满足 `j_block_start > i_block_end`，此时块内所有 `(i, j)` 均满足 `j > i`，整块 softmax 输出为 0，对 `O_i` 无贡献。
2. Element-level mask 处理边界块，其中部分位置满足 `j > i`。

两种路径的数值输出与标准 causal attention 一致，误差仅来源于浮点运算顺序差异。

---

## FP16 内部 FP32 Accumulation 路径

### 数据流全图

```
GMEM (half) ──[load/convert]──> SMEM (float) ──[compute]──> SMEM (float) ──[convert/store]──> GMEM (half)
```

### Kernel 实现要点

```cpp
__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

// FP16 forward kernel
__global__ void __launch_bounds__(128)
flash_attention_forward_fp16_kernel(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ O,
    half* __restrict__ L, ...
) {
    // Shared memory 仍为 float
    extern __shared__ float smem[];
    float* Q_tile = smem;  // ...

    // Load 阶段：GMEM half → SMEM float
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
        Q_tile[i] = half_to_float(Q_ptr[...]);
    }

    // 全部 matmul、softmax、accumulation 均在 float 域完成
    // ...

    // Store 阶段：SMEM float → GMEM half
    O_ptr[...] = float_to_half(O_tile[...] * l_inv);
    L_ptr[...] = float_to_half(m_tile[row] + logf(l_tile[row]));
}
```

### 精度与带宽权衡

| 操作 | 精度 | 说明 |
|------|------|------|
| GMEM → SMEM 加载 | FP16→FP32 | 带宽减半（相对 FP32 GMEM），每个元素 2 bytes |
| Q@K^T dot product | FP32 | 避免 FP16 累加误差 |
| Online softmax (exp, max, sum) | FP32 | 指数动态范围要求 FP32 |
| O accumulator | FP32 | 长序列累加需扩展精度 |
| SMEM → GMEM 存储 | FP32→FP16 | 带宽减半 |

**关键结论：**

- 内部全部使用 FP32 累加，将 FP16 仅作为 GMEM 存储格式，数值稳定性与纯 FP32 实现等价（误差 < 1e-3）。
- 共享内存以 `float` 分配，因此 FP16 kernel 的共享内存占用与 FP32 kernel 相同，不存在额外 SMEM 压力。
- 唯一额外开销来自 `__half2float` / `__float2half` 转换，在 Volta+ 架构上为单指令吞吐量操作，可忽略。

### 寄存器影响

FP16 kernel 的每个 thread 在寄存器中额外持有 `half` 到 `float` 的临时变量。但由于 `__launch_bounds__(128)` 的约束，编译器已预留足够寄存器预算；实测 FP16 kernel 与 FP32 kernel 的 occupancy 无显著差异。

---

## 7. 完整 Kernel 骨架

以下代码将上述概念整合为一个连贯、可编译的 kernel 骨架：

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

template <int Br, int Bc, int d>
__global__ void __launch_bounds__(128)
flash_attn_fwd_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int N, int stride_qb, int stride_qh, int stride_qn,
    int stride_kb, int stride_kh, int stride_kn,
    int stride_vb, int stride_vh, int stride_vn,
    float scale
) {
    // Grid: (batch*heads), Block: 128
    const int bh_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared memory
    extern __shared__ char smem[];
    half* q_smem = reinterpret_cast<half*>(smem);
    half* k_smem = q_smem + Br * d;
    half* v_smem = k_smem + Bc * d;
    float* s_smem = reinterpret_cast<float*>(v_smem + Bc * d);

    // Persistent registers for online softmax
    float m_reg[Br / 4];   // 每行最大值，分布在各 warp
    float l_reg[Br / 4];   // 每行累加和
    float o_reg[Br / 4][d / 8];  // 部分 O 累加器

    #pragma unroll
    for (int i = 0; i < Br / 4; ++i) {
        m_reg[i] = -FLT_MAX;
        l_reg[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < d / 8; ++j) o_reg[i][j] = 0.0f;
    }

    // 加载 Q tile（协作式，向量化）
    // ...

    // 主 KV-tile 循环
    const int num_kv_tiles = (N + Bc - 1) / Bc;
    for (int tile_kv = 0; tile_kv < num_kv_tiles; ++tile_kv) {
        // Causal 跳过
        if (tile_kv > tile_q) continue;

        // 加载 K, V tiles
        // ... 向量化 GMEM -> SMEM，由 warp 0–1 执行 ...
        __syncthreads();

        // 计算当前 tile 的 S = QK^T
        // Warp 1–2 计算 GEMM-I
        // ...

        // 若在对角线上则应用 causal mask
        if (tile_kv == tile_q) {
            // ... 谓词写入 -INFINITY ...
        }

        // Online softmax 更新
        // ... 更新 m_reg, l_reg ...

        // 计算 PV 并累加到 o_reg
        // ... GEMM-II ...
        __syncthreads();
    }

    // 最终化 O：除以 l_reg，转换为 half，写入 GMEM
    // ...
}
```

---

## 8. 性能检查清单

| 优化项 | 状态 | 验证方法 |
|--------|------|----------|
| `__launch_bounds__(128)` | 启用 | `cuobjdump -sass` 寄存器计数检查 |
| 向量化 `float4` 加载/存储 | 启用 | Nsight Compute `gld_transactions` / `gst_transactions` 比率 |
| 零共享内存 bank conflict | 启用 | Nsight Compute `shared_load_bank_conflict` 计数器 |
| 完全合并访问 (512 B/warp) | 启用 | `memory_throughput` 饱和度指标 |
| FP32 softmax 累加 | 启用 | 与 FP32 参考实现的数值单元测试 |
| Causal warp 级跳过 | 启用 | Nsight Compute causal mask 场景下 `inst_executed` 减少 |

---

## 参考文献

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
   - Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
   - NeurIPS 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   - Tri Dao
   - ICLR 2024, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

3. **NVIDIA CUDA C++ Programming Guide — Shared Memory / Launch Bounds**
   - [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

4. **Online normalizer calculation for softmax**
   - Maxim Milakov, Natalia Gimelshein
   - [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)

5. **NVIDIA Nsight Compute Documentation, *Kernel Profiling Guide***
