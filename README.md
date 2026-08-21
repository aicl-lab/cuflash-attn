# CuFlash-Attn

> 📚 Portfolio map: https://github.com/open-infra-ai/open-infra-ai

> **从零实现的 CUDA C++ FlashAttention 教学/参考实现；FP16/BF16 前向已接入 WMMA。**

[![CI](https://img.shields.io/github/actions/workflow/status/open-infra-ai/cuflash-attn/ci.yml?branch=master&style=flat-square&logo=github&label=CI)](https://github.com/open-infra-ai/cuflash-attn/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/open-infra-ai/cuflash-attn/codeql.yml?branch=master&style=flat-square&logo=github&label=CodeQL)](https://github.com/open-infra-ai/cuflash-attn/actions/workflows/codeql.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/open-infra-ai/cuflash-attn/pages.yml?branch=master&style=flat-square&logo=githubpages&logoColor=white&label=文档)](https://open-infra-ai.github.io/cuflash-attn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/github/v/release/open-infra-ai/cuflash-attn?style=flat-square&label=版本)](https://github.com/open-infra-ai/cuflash-attn/releases)

[文档](https://open-infra-ai.github.io/cuflash-attn/) · [API 参考](https://open-infra-ai.github.io/cuflash-attn/api-reference) · [更新日志](CHANGELOG.md)

---

## 🎯 项目简介

CuFlash-Attn 是一个**从零实现的 FlashAttention 算法**，用于教育学习、研究实验和轻量集成。

在五仓学习路径中，本仓库负责 CUDA C++ FlashAttention 前向/反向的专项深挖；CUDA 基础、Triton 实现、完整推理运行时和 Serving 控制面分别由其他主仓承担。整体顺序见 [`cuda-foundations/LEARNING_PATH.md`](https://github.com/open-infra-ai/cuda-foundations/blob/master/LEARNING_PATH.md)。

### 项目状态

- **状态**：稳定的 `v0.5.0` 代码库，正在做维护收敛
- **权威入口**：公开头文件、实现、测试和用户文档
- **定位**：面向学习、审计与轻量集成的精简参考实现
- **当前重点**：删除过时流程层、收紧文档并修复长尾缺陷，而不是扩展新功能

### 为什么选择 CuFlash-Attn？

| 挑战 | 解决方案 |
|------|----------|
| 📚 **学习 FlashAttention** | 清晰、文档完善的 CUDA 内核，逐步的算法实现 |
| 🔬 **研究与实验** | 修改注意力机制，无需复杂的框架依赖 |
| 🔧 **便于集成** | C++ API 配合 C ABI 绑定，可通过 ctypes 接入 Python |
| ⚡ **GPU 优化** | 多架构支持，从 V100 (sm_70) 到 H100 (sm_90) |

---

## ✨ 主要特性

| 特性 | 说明 |
|------|------|
| ⚡ **O(N) 辅助内存** | 不物化完整 O(N²) 注意力矩阵 |
| 🔢 **多精度支持** | FP32、FP16、BF16 的前向和反向路径 |
| 🔁 **完整训练** | 完整的前向/反向传播，包含梯度计算 |
| 🎭 **因果掩码** | 内置自回归模型支持（GPT 风格） |
| 🔧 **易于集成** | 简洁的 C++ API + C ABI，便于 Python ctypes 集成 |
| 🏎️ **多架构** | 优化的 CUDA 内核，支持 sm_70 → sm_90（V100 → H100） |
| 🧪 **全面测试** | 单元测试、集成测试、压力测试、PyTorch 对比测试 |
| 📊 **性能基准** | Google Benchmark 集成，用于性能追踪 |
| 📚 **完整文档** | VitePress 文档站点，覆盖算法、API、性能与故障排除 |

### 与同类库对比

| 特性 | CuFlash-Attn | PyTorch SDPA | xFormers | FlashAttention-2 |
|------|--------------|--------------|----------|------------------|
| **教学代码** | ✅ 清晰简洁 | ❌ 复杂难读 | ❌ 复杂难读 | ⚠️ 中等 |
| **自定义修改** | ✅ 容易 | ⚠️ 困难 | ⚠️ 困难 | ⚠️ 困难 |
| **无需框架依赖** | ✅ 是 | ❌ PyTorch | ❌ PyTorch | ❌ PyTorch/Cutlass |
| **Python 绑定** | ✅ ctypes | ✅ 原生 | ✅ 原生 | ✅ PyTorch |
| **训练支持** | ✅ 完整 | ✅ 完整 | ✅ 完整 | ✅ 完整 |
| **BF16 支持** | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |

> **选择 CuFlash-Attn 的场景**：希望理解、修改或嵌入 FlashAttention，同时避免繁重的依赖。

---

## 🧭 项目边界（IN / OUT）

**IN（本仓库负责）**：
- FlashAttention 前向 + 反向
- FP16 / BF16 / FP32 多精度，causal mask
- FlashDecoding（decode 阶段 KV 分块并行）
- 优化叙事与 benchmark（Roofline、kernel 深挖）

**OUT（明确不做，见对应仓库）**：
- GEMM 基础与 CUDA 编程学习 → [cuda-foundations](https://github.com/open-infra-ai/cuda-foundations)
- Triton 版算子（参考实现）→ [triton-fused-ops](https://github.com/open-infra-ai/triton-fused-ops)
- 完整推理运行时（模型加载/采样/生成）→ [tiny-llm](https://github.com/open-infra-ai/tiny-llm)

## 🚀 快速开始

### 环境要求

- **GPU**: NVIDIA GPU，计算能力 7.0+（V100、RTX 20/30/40、A100、H100）
- **CUDA Toolkit**: 11.8 或更高版本（推荐 12.x；sm_90/Hopper 需要 11.8+；注意 CUDA 13.x 已移除 sm_70 支持）
- **CMake**: 3.18 或更高版本（使用 CMake preset 需 3.20+）
- **编译器**: GCC 7+、Clang 5+ 或 MSVC 2017+（需要 C++17 支持）

### 方案 1：源码编译

```bash
# 克隆仓库
git clone https://github.com/open-infra-ai/cuflash-attn.git
cd cuflash-attn

# 使用预设构建（Release 模式）
cmake --preset release
cmake --build --preset release

# 运行测试
ctest --preset release --output-on-failure
```

### 方案 2：Docker（推荐用于快速测试）

```bash
# 构建 Docker 镜像
docker build -t cuflash-attn .

# 运行（需要 GPU 支持）
docker run --gpus all -it cuflash-attn

# 容器内运行基准测试
./cuflash_attn_bench
```

### 第一个 C++ 程序

```cpp
#include <cuda_runtime.h>
#include "cuflash/flash_attention.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// 辅助函数：用随机值初始化设备内存
void init_random(float* d_ptr, size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> h_data(n);
    for (auto& v : h_data) v = dist(gen);
    cudaMemcpy(d_ptr, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

int main() {
    const int B = 2, H = 8, N = 1024, D = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const size_t qkv_size = B * H * N * D;
    const size_t l_size = B * H * N;
    
    // 分配设备内存
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));
    
    // 用随机数据初始化
    init_random(d_Q, qkv_size, 1);
    init_random(d_K, qkv_size, 2);
    init_random(d_V, qkv_size, 3);
    
    // 使用因果掩码计算 FlashAttention
    auto err = cuflash::flash_attention_forward(
        d_Q, d_K, d_V, d_O, d_L,
        B, H, N, D, scale,
        true  // 因果掩码
    );
    
    if (err != cuflash::FlashAttentionError::SUCCESS) {
        std::cerr << "错误: " << cuflash::get_error_string(err) << std::endl;
        return 1;
    }
    
    std::cout << "FlashAttention 前向传播成功完成！" << std::endl;
    
    // 释放资源
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
    return 0;
}
```

📖 **更多示例**: 请参见 [examples/](examples/) 目录，包含反向传播和 Python 集成的完整程序。

### Python 集成

```python
import ctypes
import numpy as np
import torch

# 加载动态库
lib = ctypes.CDLL("./build/release/libcuflash_attn.so")

# 定义 API
lib.cuflash_attention_forward_f32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_bool, ctypes.c_void_p
]
lib.cuflash_attention_forward_f32.restype = ctypes.c_int

# 使用 PyTorch 准备数据
B, H, N, D = 2, 8, 1024, 64
Q = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
K = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
V = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
O = torch.empty_like(Q)
L = torch.empty(B, H, N, dtype=torch.float32, device='cuda')

# 调用 CuFlash-Attn
scale = 1.0 / np.sqrt(D)
result = lib.cuflash_attention_forward_f32(
    ctypes.c_void_p(Q.data_ptr()),
    ctypes.c_void_p(K.data_ptr()),
    ctypes.c_void_p(V.data_ptr()),
    ctypes.c_void_p(O.data_ptr()),
    ctypes.c_void_p(L.data_ptr()),
    B, H, N, D, scale, True, None
)

assert result == 0, f"FlashAttention 失败，错误码 {result}"
print(f"输出形状: {O.shape}, 平均值: {O.mean().item():.4f}")
```

📖 **完整 Python 示例**: 参见 [examples/python_binding.py](examples/python_binding.py)

---

## 📊 性能

FlashAttention 通过分块与在线 softmax 避免物化完整注意力矩阵。实际延迟、吞吐和显存收益取决于 GPU、CUDA、dtype、形状和 causal 模式；仓库不维护无法持续复测的固定性能数字，请在目标硬件上运行自带 benchmark。

**指标口径**：FlashAttention kernel 用 **CUDA Event 计时**（排除 host 开销）；
带宽指标为**模型化 logical HBM**（前向：Q/O 各一次 + K/V 按 Q block 重载，见
[algorithm](../docs/algorithm.md)），并非 ncu 实测的物理带宽。请勿把 `LogicalHBM GB/s`
当作硬件实测 HBM 带宽。

### 运行基准测试

```bash
cmake --preset release
cmake --build --preset release
./build/release/cuflash_attn_bench
```

---

## 📖 文档

### 快速链接

| 资源 | 链接 |
|------|------|
| 📘 **完整文档** | [https://open-infra-ai.github.io/cuflash-attn/](https://open-infra-ai.github.io/cuflash-attn/) |
| 🔌 **API 参考** | [API 文档](https://open-infra-ai.github.io/cuflash-attn/api-reference) |
| 🧠 **算法详解** | [深入理解 FlashAttention](https://open-infra-ai.github.io/cuflash-attn/algorithm) |
| 🔧 **构建指南** | [从源码构建](https://open-infra-ai.github.io/cuflash-attn/building) |
| ❓ **故障排除** | [常见问题与解决方案](https://open-infra-ai.github.io/cuflash-attn/troubleshooting) |

---

## ⚙️ 配置

### 支持的参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `head_dim` | 32, 64, 128 | 内核优化必需 |
| **数据类型** | FP32 (`float`), FP16 (`half`), BF16 (`__nv_bfloat16`) | 前向和反向都支持 |
| **因果掩码** | 可选 | 运行时启用/禁用 |
| **批大小** | ≥ 1 | 任意正整数 |
| **序列长度** | ≥ 1 | 优化用于 1K-16K+ |
| **头数** | ≥ 1 | 任意正整数 |

### GPU 架构支持

| 架构 | 计算能力 | 示例 GPU |
|------|---------|----------|
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080 Ti |
| Ampere | sm_80, sm_86 | A100, RTX 3090 |
| Ada Lovelace | sm_89 | RTX 4090 |
| Hopper | sm_90 | H100 |

**默认构建目标**: sm_80, sm_86（A100 + RTX 30xx）。可配置编译 sm_70–sm_90；
仓库当前实测硬件为 **sm_86（RTX 3060）**。

自定义使用: `cmake --preset release -DCMAKE_CUDA_ARCHITECTURES="90"`

---

## 🏗️ 项目结构

```
cuflash-attn/
├── benchmarks/                 # 性能基准测试（Google Benchmark）
├── cmake/                      # CMake 模块和打包配置
├── docs/                       # VitePress 文档站点（中文）
│   ├── .vitepress/             # 站点配置与主题
│   ├── public/                 # 静态资源（logo、favicon）
│   ├── guide/                  # 快速开始指南
│   ├── design/                 # 设计决策与内核深度解读
│   ├── performance/            # 基准测试与 Roofline 分析
│   └── research/               # 相关工作与参考文献
├── examples/                   # 完整使用示例
│   ├── basic_usage.cu          # C++ 基础示例
│   └── python_binding.py       # Python ctypes 示例
├── include/cuflash/            # 公共 API 头文件
│   ├── flash_attention.h       # 主 API，包含 C++ 和 C ABI
│   ├── export.h                # 可见性宏
│   └── version.h.in            # 版本头文件模板
├── src/                        # 实现代码
│   ├── api/                    # API 调度层
│   ├── forward/                # 前向传播内核实现
│   ├── backward/               # 反向传播内核实现
│   └── kernels/                # 内部内核工具（.cuh）
├── tests/                      # 测试套件
│   ├── unit/                   # 单元测试（10 个文件）
│   ├── integration/            # 集成测试 + PyTorch 对比
│   └── package_smoke/          # 包冒烟测试
├── CMakeLists.txt              # 主构建配置
├── CMakePresets.json           # 构建预设（release、debug、asan）
├── Dockerfile                  # 容器构建
└── .github/workflows/          # CI/CD 工作流
    ├── ci.yml                  # 多配置构建矩阵与测试
    ├── codeql.yml              # 安全扫描
    ├── pages.yml               # 文档部署
    └── release.yml             # 发布自动化
```

---

## 🧪 测试与质量

### 测试分类

```bash
# 运行所有测试
ctest --preset release --output-on-failure

# 运行特定测试类别
ctest --preset release -R ForwardTest    # 前向传播测试
ctest --preset release -R BackwardTest   # 反向传播测试
ctest --preset release -R StressTest     # 压力与边界测试
ctest --preset release -R PyTorch        # PyTorch 对比测试（需要 GPU + PyTorch）
```

### 代码质量工具

- ✅ **clang-format**: 自动化代码格式化（CI 强制执行）
- ✅ **clang-tidy**: 静态分析，50+ 检查（host 源文件在 CI 中强制执行）
- ✅ **CodeQL**: 每周安全扫描
- ✅ **Sanitizers**: AddressSanitizer & UBSan（CI 构建矩阵）+ compute-sanitizer（GPU workflow）

```bash
# 使用 AddressSanitizer 构建
cmake --preset debug-asan
cmake --build --preset debug-asan
ctest --preset debug-asan
```

---

## 🤝 贡献

欢迎贡献。请保持改动聚焦、明确，并围绕当前 CUDA 库边界推进。

### 开始贡献

1. **克隆仓库** 并确认当前范围
2. **使用 preset 构建**，先验证环境可用
3. **编写或更新测试** 覆盖你的改动
4. **同步更新文档**，确保行为、API 用法和工作流描述一致
5. **提交拉取请求**，附上简明摘要和验证说明

### 开发工作流

```bash
# 提交前格式化代码
find . -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# 本地运行测试
cmake --preset release && cmake --build --preset release
ctest --preset release --output-on-failure

# 可选：运行 clang-tidy（host 源文件）
./scripts/run_clang_tidy.sh build/release
```

📋 **详细指南**: 请参见 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 📚 参考文献

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao 等，NeurIPS 2022
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao，ICLR 2024

---

## 📈 版本历史

详细的版本历史和更新请参见 [CHANGELOG.md](CHANGELOG.md)。

---

<p align="center">
  <sub>用 ❤️ 打造的教学/参考注意力实现</sub><br>
  <sub>精简参考实现 · CUDA C++ · 开源</sub>
</p>
