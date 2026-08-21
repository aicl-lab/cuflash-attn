# 快速开始

几分钟内让 CuFlash-Attn 运行起来。

## 前置条件

- NVIDIA GPU，计算能力 7.0+ (V100, RTX 20/30/40 系列, A100, H100)
- CUDA Toolkit 11.8 或更高版本（推荐 12.x；sm_90 需 11.8+；CUDA 13.x 已移除 sm_70）
- CMake 3.18 或更高版本
- C++17 兼容编译器

## 安装

### 克隆仓库

```bash
git clone https://github.com/open-infra-ai/cuflash-attn.git
cd cuflash-attn
```

### 使用 CMake Presets 构建

```bash
# 配置（Release 构建）
cmake --preset release

# 构建
cmake --build --preset release

# 运行测试
ctest --preset release --output-on-failure
```

### 自定义 Preset 覆盖参数

```bash
# Release 构建并指定自定义架构目标
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build --preset release -j$(nproc)
```

## 你的第一个程序

```cpp
#include <cuda_runtime.h>
#include "cuflash/flash_attention.h"
#include <iostream>
#include <cmath>

int main() {
    // 配置
    const int B = 2, H = 8, N = 1024, D = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const bool causal = true;

    // 分配和初始化张量（简化版）
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, B * H * N * D * sizeof(float));
    cudaMalloc(&d_K, B * H * N * D * sizeof(float));
    cudaMalloc(&d_V, B * H * N * D * sizeof(float));
    cudaMalloc(&d_O, B * H * N * D * sizeof(float));
    cudaMalloc(&d_L, B * H * N * sizeof(float));

    // 用你的数据初始化 Q, K, V...

    // 计算 FlashAttention
    auto err = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L,
        B, H, N, D, scale, causal
    );

    if (err != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "错误: " << cuflash::get_error_string(err) << std::endl;
        return 1;
    }

    // d_O 现在包含输出

    // 清理
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);

    return 0;
}
```

## 下一步

- 查看 [API 参考](/api-reference)
- 阅读 [算法详解](/algorithm)
- 了解 [构建选项](/building)
