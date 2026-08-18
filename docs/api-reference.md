# API 参考

CuFlash-Attn 提供简洁的 C++ API，所有函数和类型定义在 `cuflash` 命名空间中。

---

## 目录

- [头文件](#头文件)
- [前向传播](#前向传播)
  - [FP32 前向](#flash_attention_forward-fp32)
  - [FP16 前向](#flash_attention_forward-fp16)
  - [BF16 前向](#flash_attention_forward-bf16)
- [反向传播](#反向传播)
  - [FP32 反向](#flash_attention_backward-fp32)
  - [FP16 反向](#flash_attention_backward-fp16)
  - [BF16 反向](#flash_attention_backward-bf16)
- [C ABI 接口](#c-abi-接口)
- [张量布局](#张量布局)
- [错误处理](#错误处理)
- [类型支持](#类型支持)
- [构建选项](#构建选项)
- [GPU 架构支持](#gpu-架构支持)

---

## 头文件

```cpp
#include "cuflash/flash_attention.h"
```

所有公共 API 均通过此单一头文件暴露。**`include/cuflash/flash_attention.h` 是唯一权威
签名来源**；本文档若与头文件不一致，以头文件为准。

> 完整的前向/反向签名覆盖 FP32 (`float`)、FP16 (`half`)、BF16 (`__nv_bfloat16`)
> 三种 dtype，本文档按 dtype 分节列出。

---

## 前向传播

### `flash_attention_forward` (FP32)

计算 FP32 精度的 FlashAttention 前向传播。

```cpp
FlashAttentionError flash_attention_forward(
    const float* Q,          // 查询张量 [B, H, N, D]
    const float* K,          // 键张量 [B, H, N, D]
    const float* V,          // 值张量 [B, H, N, D]
    float* O,                // 输出张量 [B, H, N, D]
    float* L,                // logsumexp [B, H, N]（反向传播需要）
    int batch_size,          // 批大小 B
    int num_heads,           // 注意力头数 H
    int seq_len,             // 序列长度 N
    int head_dim,            // 头维度 D（32、64 或 128）
    float scale,             // 缩放因子，通常 1.0f / sqrt(D)
    bool causal,             // 是否启用因果掩码
    cudaStream_t stream = 0  // CUDA 流（0 表示默认流）
);
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `Q` | `const float*` | 设备内存中的查询张量 |
| `K` | `const float*` | 设备内存中的键张量 |
| `V` | `const float*` | 设备内存中的值张量 |
| `O` | `float*` | 设备内存中的输出张量 |
| `L` | `float*` | 设备内存中的 logsumexp 值 |
| `batch_size` | `int` | 批次中的序列数量 |
| `num_heads` | `int` | 注意力头数 |
| `seq_len` | `int` | 输入序列长度 |
| `head_dim` | `int` | 每个头的维度（32、64 或 128） |
| `scale` | `float` | 注意力缩放因子 |
| `causal` | `bool` | 是否应用因果（自回归）掩码 |
| `stream` | `cudaStream_t` | 异步执行的 CUDA 流 |

**返回值：** 成功时返回 `FlashAttentionError::SUCCESS`，否则返回错误代码。

---

### `flash_attention_forward` (FP16)

计算 FP16 精度的 FlashAttention 前向传播。内部计算使用 FP32 以确保数值稳定性，输出转换回 FP16。

```cpp
FlashAttentionError flash_attention_forward(
    const half* Q,           // 查询张量 [B, H, N, D]
    const half* K,           // 键张量 [B, H, N, D]
    const half* V,           // 值张量 [B, H, N, D]
    half* O,                 // 输出张量 [B, H, N, D]
    float* L,                // logsumexp [B, H, N]（始终为 FP32）
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

**精度处理：**
- 输入/输出：FP16（16 位半精度）
- 内部计算：FP32（32 位单精度）
- 最终结果：FP16

此方法在减少内存带宽需求的同时，提供了与 FP32 相当的数值稳定性。
在支持架构（sm_70+）上前向走 WMMA（Tensor Core）路径。

---

### `flash_attention_forward` (BF16)

计算 BF16 精度的 FlashAttention 前向传播。内部计算使用 FP32 以确保数值稳定性，输出转换回 BF16。

```cpp
FlashAttentionError flash_attention_forward(
    const __nv_bfloat16* Q,      // 查询张量 [B, H, N, D]
    const __nv_bfloat16* K,      // 键张量 [B, H, N, D]
    const __nv_bfloat16* V,      // 值张量 [B, H, N, D]
    __nv_bfloat16* O,            // 输出张量 [B, H, N, D]
    float* L,                    // logsumexp [B, H, N]（始终为 FP32）
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

**精度处理：**
- 输入/输出：BF16（`__nv_bfloat16`）
- 内部计算：FP32（32 位单精度）
- 最终结果：BF16

在支持架构（sm_80+）上前向走 WMMA（Tensor Core）路径。

---

## 反向传播

### `flash_attention_backward` (FP32)

计算 FP32 精度的 FlashAttention 反向传播梯度。

```cpp
FlashAttentionError flash_attention_backward(
    const float* Q,          // 前向传播的查询张量
    const float* K,          // 前向传播的键张量
    const float* V,          // 前向传播的值张量
    const float* O,          // 前向传播的输出张量
    const float* L,          // 前向传播的 logsumexp
    const float* dO,         // 上游梯度 [B, H, N, D]
    float* dQ,               // Q 的梯度（输出）
    float* dK,               // K 的梯度（输出）
    float* dV,               // V 的梯度（输出）
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

**梯度计算：**
- 使用重计算策略（反向传播期间重新计算注意力权重）
- 不存储 O(N²) 的注意力矩阵
- 内存复杂度：O(N) 而非 O(N²)

**要求：**
- `O` 和 `L` 必须来自相应的前向传播调用
- `dQ`、`dK`、`dV` 必须在设备内存中预先分配

---

### `flash_attention_backward` (FP16)

计算 FP16 精度的 FlashAttention 反向传播梯度。

```cpp
FlashAttentionError flash_attention_backward(
    const half* Q,
    const half* K,
    const half* V,
    const half* O,
    const float* L,
    const half* dO,
    half* dQ,
    half* dK,
    half* dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

**实现说明：**
- 内部累加使用 FP32 以防止溢出
- 最终梯度转换为 FP16
- 数值稳定性与 FP32 反向传播相当

---

### `flash_attention_backward` (BF16)

计算 BF16 精度的 FlashAttention 反向传播梯度（scalar 路径）。

```cpp
FlashAttentionError flash_attention_backward(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const __nv_bfloat16* O,
    const float* L,
    const __nv_bfloat16* dO,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool causal,
    cudaStream_t stream = 0
);
```

**实现说明：**
- 内部累加使用 FP32 以防止溢出
- 最终梯度转换为 BF16
- 数值稳定性与 FP32 反向传播相当
- 反向目前全部为 scalar 路径（backward WMMA 不纳入 v1.0）

---

## C ABI 接口

用于通过 `ctypes` 或其他语言调用的 C 兼容函数。

### FP32 接口

```c
// 前向传播 - C ABI
int cuflash_attention_forward_f32(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);

// 反向传播 - C ABI
int cuflash_attention_backward_f32(
    const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* dO,
    float* dQ, float* dK, float* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);
```

### FP16 接口

```c
// 前向传播 - C ABI
int cuflash_attention_forward_f16(
    const half* Q, const half* K, const half* V,
    half* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);

// 反向传播 - C ABI
int cuflash_attention_backward_f16(
    const half* Q, const half* K, const half* V,
    const half* O, const float* L, const half* dO,
    half* dQ, half* dK, half* dV,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream
);
```

**返回值：** `FlashAttentionError` 枚举的整数值。

---

## 张量布局

所有张量使用**行优先（C 风格）**内存布局。

### 张量形状

| 张量 | 形状 | 说明 |
|------|------|------|
| `Q`、`K`、`V`、`O` | `[batch_size, num_heads, seq_len, head_dim]` | 输入/输出张量 |
| `dQ`、`dK`、`dV`、`dO` | `[batch_size, num_heads, seq_len, head_dim]` | 梯度张量 |
| `L` | `[batch_size, num_heads, seq_len]` | logsumexp 值 |

### 内存偏移计算

```cpp
// 访问 Q[b][h][s][d]
size_t offset = ((b * num_heads + h) * seq_len + s) * head_dim + d;

// 访问 L[b][h][s]
size_t offset = (b * num_heads + h) * seq_len + s;
```

### 数据类型详情

- **float**：32 位 IEEE 754 单精度浮点数
- **half**：16 位 IEEE 754 半精度浮点数（CUDA 原生）
- **`__nv_bfloat16`**：16 位 Brain Float（CUDA 原生，`cuda_bf16.h`）
- 所有指针必须指向连续的设备内存

---

## 错误处理

### `FlashAttentionError` 枚举

```cpp
enum class FlashAttentionError {
    SUCCESS = 0,                   // 操作成功完成
    INVALID_DIMENSION,             // 维度参数无效（≤ 0）
    DIMENSION_MISMATCH,            // 预留，将来使用
    NULL_POINTER,                  // 输入或输出指针为空
    CUDA_ERROR,                    // CUDA 运行时错误
    OUT_OF_MEMORY,                 // GPU 显存不足
    UNSUPPORTED_HEAD_DIM,          // head_dim 不在 {32, 64, 128} 中
    UNSUPPORTED_DTYPE              // 不支持的数据类型
};
```

### `get_error_string`

```cpp
const char* get_error_string(FlashAttentionError error);
```

返回错误代码对应的人类可读字符串。

### 错误处理示例

```cpp
#include "cuflash/flash_attention.h"
#include <iostream>

int main() {
    // ... 为 d_Q、d_K、d_V、d_O、d_L 分配设备内存 ...
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    auto err = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L,
        batch_size, num_heads, seq_len, head_dim,
        scale,
        /*causal=*/true
    );
    
    if (err != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "FlashAttention 错误: "
                  << cuflash::get_error_string(err) << std::endl;
        return 1;
    }
    
    // 反向传播
    err = cuflash::flash_attention_backward(
        d_Q, d_K, d_V, d_O, d_L, d_dO,
        d_dQ, d_dK, d_dV,
        batch_size, num_heads, seq_len, head_dim,
        scale, true
    );
    
    if (err != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "反向传播错误: "
                  << cuflash::get_error_string(err) << std::endl;
        return 1;
    }
    
    return 0;
}
```

---

## 类型支持

### 支持的配置

| 参数 | 支持的值 |
|------|----------|
| `head_dim` | 32、64、128 |
| 数据类型 | `float` (FP32)、`half` (FP16)、`__nv_bfloat16` (BF16) |
| 因果掩码 | 可选（`bool causal`） |
| 批大小 | ≥ 1 |
| 注意力头数 | ≥ 1 |
| 序列长度 | ≥ 1 |

### 数据类型支持矩阵

| 数据类型 | 前向传播 | 反向传播 |
|----------|----------|----------|
| `float` (FP32) | ✅ 完全支持（scalar） | ✅ 完全支持（scalar） |
| `half` (FP16) | ✅ 完全支持（sm_70+ WMMA / scalar fallback） | ✅ 完全支持（scalar） |
| `__nv_bfloat16` (BF16) | ✅ 完全支持（sm_80+ WMMA / scalar fallback） | ✅ 完全支持（scalar） |

---

## 构建选项

| CMake 选项 | 默认值 | 说明 |
|------------|--------|------|
| `BUILD_TESTS` | ON | 构建 GoogleTest 测试套件 |
| `ENABLE_RAPIDCHECK` | OFF | 启用 RapidCheck 基于属性的测试 |
| `BUILD_SHARED_LIBS` | ON | 构建为共享库（`*.so`/`.dll`/`.dylib`） |
| `BUILD_EXAMPLES` | ON | 构建示例程序 |
| `ENABLE_FAST_MATH` | OFF | 启用 `--use_fast_math` 编译器标志 |

### 示例配置

```bash
# 高性能发布版本构建
cmake --preset release-fast-math \
      -DBUILD_SHARED_LIBS=OFF
cmake --build --preset release-fast-math

# 带所有测试的调试版本
cmake --preset default \
      -DENABLE_RAPIDCHECK=ON
cmake --build --preset default

# 仅静态库
cmake --preset minimal \
      -DBUILD_SHARED_LIBS=OFF
cmake --build --preset minimal
```

---

## GPU 架构支持

### 支持的 CUDA 架构

| 架构 | 计算能力 | 代表 GPU |
|------|----------|----------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80、sm_86 | A100、RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |

### 架构特定调优

默认构建支持所有架构。针对特定部署：

```bash
# 仅支持 RTX 3090 / A100
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=86

# 支持多个架构
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
```

### 共享内存需求（scalar 前向，FP32）

| head_dim | SRAM 需求 | 典型块大小 |
|----------|-----------|-----------|
| 32 | ~48 KB | 64 × 64 |
| 64 | ~80 KB | 64 × 64 |
| 128 | ~68 KB | 32 × 32 |

注意：hd64/hd128 超过默认 48 KB 动态共享内存上限，launcher 通过
`cudaFuncSetAttribute` 申请 opt-in 动态共享内存（sm_86 上限 100 KB）。
FP16/BF16 的 WMMA 前向 tile 更小（hd64 为 64×32）。

---

## 线程安全

### 前向传播
- 使用不同流并发调用时完全线程安全
- 无共享可变状态

### 反向传播
- 使用内部静态工作空间进行中间缓冲区分配
- 对于单线程 CUDA context 使用（常见情况）是**线程安全的**
- 对于多主机线程并发调用反向传播**非线程安全**

#### 多流并发使用

对于多流并发执行反向传播，有两种选择：

1. **每线程顺序执行**（推荐）：
   ```cpp
   // 安全：每个线程在自己的流上顺序调用
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   
   // 线程 1：在 stream1 上顺序调用
   flash_attention_backward(..., stream1);
   flash_attention_backward(..., stream1);  // 安全
   
   // 线程 2：需要与线程 1 同步
   ```

2. **线程间同步**：
   - 在另一个线程调用反向传播前使用 `cudaStreamSynchronize()`
   - 或对反向传播调用使用外部互斥锁

::: warning 参考实现说明
此设计适用于教育和单流生产场景。
对于多线程训练流水线，建议外部管理工作空间。
:::

## 内存管理

- 所有张量分配由调用者负责
- 内核执行期间不进行动态内存分配
- 工作空间内存由内部使用流安全的分配管理
