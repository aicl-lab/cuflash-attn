# 故障排除指南

使用 CuFlash-Attn 时的常见问题和解决方案。

---

## 目录

- [构建问题](#构建问题)
- [运行时错误](#运行时错误)
- [性能问题](#性能问题)
- [数值精度](#数值精度)
- [错误代码参考](#错误代码参考)
- [获取帮助](#获取帮助)

---

## 构建问题

### CMake 找不到 CUDA

**症状：**
```
CMake Error: Could not find CUDA
```

**解决方案：**

1. **验证 CUDA 安装：**
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. **显式设置 CUDA 路径：**
   ```bash
   cmake --preset release \
         -DCUDAToolkit_ROOT=/usr/local/cuda \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
   ```

3. **常见路径位置：**
   | 操作系统 | 默认 CUDA 路径 |
   |---|-------------------|
   | Linux | `/usr/local/cuda` |
   | Windows | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8` |

4. **设置环境变量：**
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

---

### 不支持的 GPU 架构

**症状：**
```
nvcc fatal : Unsupported GPU architecture 'sm_89'
```

**原因：** CUDA toolkit 版本不支持您的 GPU 架构。

**解决方案：**

1. **更新 CUDA Toolkit**（对于新 GPU 推荐）

2. **定位兼容的架构：**
   ```bash
   # 检查支持的架构
   nvcc --help | grep "gpu-architecture"
   
   # 为支持的架构构建
   cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=80  # 根据您的 CUDA 版本调整
   ```

3. **兼容性矩阵：**
   | 架构 | 最低 CUDA 版本 |
   |--------------|---------------------|
   | sm_70 (V100) | CUDA 9.0 |
   | sm_75 (Turing) | CUDA 10.0 |
   | sm_80 (A100) | CUDA 11.0 |
   | sm_86 (RTX 3090) | CUDA 11.1 |
   | sm_89 (RTX 4090) | CUDA 11.8 |
   | sm_90 (H100) | CUDA 12.0 |

---

### 构建期间内存不足

**症状：**
```
nvcc fatal : Memory allocation failure
```

**解决方案：**

1. **减少并行任务：**
   ```bash
   cmake --build --preset release -j2  # 仅使用 2 个并行任务
   ```

2. **减少目标架构：**
   ```bash
   # 单架构构建
   cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=86
   ```

3. **关闭其他应用程序：** 释放系统内存

---

### 链接器错误

**症状：**
```
undefined reference to `cuflash::flash_attention_forward'
```

**解决方案：**

1. **验证库构建：**
   ```bash
   ls -la build/release/libcuflash_attn*
   ```

2. **设置库路径：**
   ```bash
   export LD_LIBRARY_PATH=$PWD/build/release:$LD_LIBRARY_PATH
   ```

3. **在您的项目中正确链接：**
   ```cmake
   target_link_libraries(your_target cuflash_attn)
   ```

---

## 运行时错误

### CUDA 内存不足

**错误代码：** `FlashAttentionError::OUT_OF_MEMORY`

**症状：**
```
CUDA error: out of memory
```

**解决方案：**

1. **检查可用内存：**
   ```bash
   nvidia-smi
   ```

2. **减小批大小或序列长度：**
   | 配置 | 内存影响 |
   |---------------|---------------|
   | batch_size | 线性 |
   | seq_len | 线性 |
   | num_heads | 线性 |
   | head_dim | 固定（32/64/128） |

3. **内存估算公式：**
   ```
   前向: ~4 × batch × heads × seq_len × head_dim (FP32 字节)
   反向: ~6 × batch × heads × seq_len × head_dim (FP32 字节)
   ```

4. **运行前释放 GPU 内存：**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### 不支持的 head 维度

**错误代码：** `FlashAttentionError::UNSUPPORTED_HEAD_DIM`

**有效值：** 32、64、128

**解决方案：**

1. **检查您的配置：**
   ```cpp
   if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
       // 不支持
   }
   ```

2. **变通方案：**
   - 填充到最近的受支持维度
   - 对不同的头大小使用多次调用

---

### 无效的维度参数

**错误代码：** `FlashAttentionError::INVALID_DIMENSION`

**原因：** batch_size、num_heads、seq_len 或 head_dim ≤ 0

**解决方案：** 验证所有维度参数都是正整数。

---

### 空指针

**错误代码：** `FlashAttentionError::NULL_POINTER`

**原因：** 一个或多个输入/输出指针是 `nullptr`

**解决方案：** 验证所有张量指针都已正确分配：
```cpp
cudaMalloc(&d_Q, batch_size * num_heads * seq_len * head_dim * sizeof(float));
// ... 分配所有必需的张量
```

---

### CUDA 运行时错误

**错误代码：** `FlashAttentionError::CUDA_ERROR`

**常见原因：**

1. **无效的内存访问：**
   - 指针未在设备内存中分配
   - 来自其他操作的内存损坏

2. **内核启动失败：**
   - 对于 GPU 来说线程或块太多
   - 资源冲突

3. **调试步骤：**
   ```bash
   # 启用 CUDA 错误检查
   export CUDA_LAUNCH_BLOCKING=1
   
   # 使用 cuda-memcheck 运行
   cuda-memcheck ./your_program
   
   # 使用 compute-sanitizer (CUDA 11+)
   compute-sanitizer ./your_program
   ```

---

## 性能问题

### 比预期慢

**诊断步骤：**

1. **验证 GPU 利用率：**
   ```bash
   nvidia-smi -l 1  # 监控 GPU 使用率
   ```

2. **检查架构匹配：**
   - 确保二进制文件是为您的 GPU 架构编译的
   - 使用正确的 `CMAKE_CUDA_ARCHITECTURES` 重新构建

3. **启用快速数学（如果精度允许）：**
   ```bash
   cmake --preset release-fast-math
   cmake --build --preset release-fast-math
   ```

4. **分析内核执行：**
   ```bash
   # 使用 Nsight Compute
   ncu ./your_program
   
   # 使用 Nsight Systems
   nsys profile ./your_program
   ```

---

### 高内存使用

**注意：** 对于 head_dim=128，由于更大的共享内存需求，这是预期的。

**优化选项：**

1. **使用 FP16 而非 FP32：**
   - 内存使用减半
   - 对大多数应用的精度影响最小

2. **减小块大小（如果自定义）：**
   - 更小的块 → 每块更少的共享内存
   - 可能影响性能

---

## 数值精度

### 结果与 PyTorch 不同

**预期的差异：**

| 方面 | 预期行为 |
|--------|-------------------|
| 小的数值差异 | FP32 ±1e-5，FP16 ±1e-3 |
| FP16 累加 | 由于舍入而方差更大 |

**诊断步骤：**

1. **检查缩放因子：**
   ```cpp
   float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
   // 应与 PyTorch 默认值匹配
   ```

2. **验证因果掩码：**
   - 确保 `causal` 参数在实现之间匹配

3. **使用相同的数据类型：**
   - FP32 与 FP32 对比
   - FP16 与 FP16 对比

---

### 输出中的 INF/NaN

**原因：**

1. **输入包含 INF/NaN：**
   ```python
   import torch
   assert not torch.isnan(Q).any()
   assert not torch.isinf(Q).any()
   ```

2. **极端的 QK 值：**
   - 如果序列是自回归的，启用因果掩码
   - 检查缩放因子计算

3. **FP16 溢出：**
   - 对有问题的输入使用 FP32
   - 在训练中启用梯度裁剪

---

### 反向传播中的梯度不匹配

**常见原因：**

1. **缺少 Logsumexp (L)：**
   - 必须将前向传播的 L 传递给反向传播
   - L 必须来自同一次前向调用

2. **错误的梯度流：**
   - dO 应该是相对于 O（前向输出）的梯度
   - dQ、dK、dV 是输出参数

---

## 错误代码参考

| 错误代码 | 值 | 含义 | 解决方案 |
|------------|-------|---------|------------|
| `SUCCESS` | 0 | 操作成功 | 无 |
| `INVALID_DIMENSION` | 1 | 维度 ≤ 0 | 检查所有维度参数 |
| `DIMENSION_MISMATCH` | 2 | 预留 | 当前未使用 |
| `NULL_POINTER` | 3 | 传递了空指针 | 验证所有分配 |
| `CUDA_ERROR` | 4 | CUDA 运行时错误 | 检查 CUDA 上下文和内存 |
| `OUT_OF_MEMORY` | 5 | GPU 内存不足 | 减小问题规模或释放内存 |
| `UNSUPPORTED_HEAD_DIM` | 6 | head_dim 不在 {32,64,128} 中 | 使用支持的维度 |
| `UNSUPPORTED_DTYPE` | 7 | 不支持的数据类型 | 使用 float 或 half |

```cpp
// 全面的错误处理示例
auto err = cuflash::flash_attention_forward(...);
switch (err) {
    case cuflash::FlashAttentionError::SUCCESS:
        break;
    case cuflash::FlashAttentionError::OUT_OF_MEMORY:
        std::cerr << "GPU 内存不足。请尝试减小批大小。\n";
        break;
    case cuflash::FlashAttentionError::UNSUPPORTED_HEAD_DIM:
        std::cerr << "head_dim 必须是 32、64 或 128。\n";
        break;
    default:
        std::cerr << "错误: " << cuflash::get_error_string(err) << "\n";
}
```

---

## 获取帮助

### 提问之前

1. **检查错误代码：** 使用 `get_error_string()` 获取详细消息
2. **验证设置：** 运行 `nvidia-smi` 和 `nvcc --version`
3. **测试基本功能：** 使用 `ctest` 运行内置测试

### 报告问题

报告问题时，请包含：

1. **系统信息：**
   ```bash
   nvidia-smi
   nvcc --version
   cmake --version
   ```

2. **错误输出：** 完整的错误消息和堆栈跟踪
3. **最小重现：** 触发问题的小代码片段
4. **构建配置：** 使用的 CMake cache 或 preset

### 资源

- [API 参考](api-reference.md)
- [构建指南](building.md)
- [算法文档](algorithm.md)
- [GitHub Issues](https://github.com/open-infra-ai/cuflash-attn/issues)
