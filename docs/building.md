# 构建指南

从源代码构建 CuFlash-Attn 的完整指南。

---

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [CMake Presets](#cmake-presets)
- [手动构建](#手动构建)
- [构建选项](#构建选项)
- [运行测试](#运行测试)
- [GPU 架构配置](#gpu-架构配置)
- [跨平台说明](#跨平台说明)
- [故障排除](#故障排除)

---

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| **CUDA Toolkit** | 11.8 | 包含 nvcc 编译器和 CUDA 库（推荐 12.x；sm_90 需 11.8+；CUDA 13.x 移除了 sm_70（Volta），如需 V100 支持请用 CUDA 12.x 构建） |
| **CMake** | 3.18（preset 需 3.20） | 构建系统生成器 |
| **C++ 编译器** | C++17 | GCC 7+、Clang 5+、MSVC 2017+ |
| **Python**（可选） | 3.8+ | 用于 PyTorch 对比测试 |
| **PyTorch**（可选） | 2.0+ | 用于验证测试 |

### 验证环境

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 CMake 版本
cmake --version

# 检查 C++ 编译器版本（Linux）
g++ --version
```

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/AICL-Lab/cuflash-attn.git
cd cuflash-attn

# 使用 preset 构建（推荐）
cmake --preset release
cmake --build --preset release

# 运行测试
ctest --preset release --output-on-failure
```

---

## CMake Presets

项目提供了预定义的 CMake Presets，用于常见的构建配置：

| Preset | 构建类型 | 测试 | 用途 |
|--------|----------|------|------|
| `default` | Debug | ✅ | 开发和调试 |
| `release` | Release | ✅ | 生产使用 |
| `release-fast-math` | Release | ✅ | 最大性能（精度略降） |
| `minimal` | Release | ❌ | 最小构建体积 |

### Preset 构建命令

```bash
# Debug 构建（含测试和示例）
cmake --preset default
cmake --build --preset default

# 优化发布版本构建
cmake --preset release
cmake --build --preset release

# Release + 快速数学优化
cmake --preset release-fast-math
cmake --build --preset release-fast-math

# 最小化构建（不含测试和示例）
cmake --preset minimal
cmake --build --preset minimal
```

---

## 使用 Preset 覆盖自定义配置

对于自定义配置，继续使用 preset，并显式覆盖缓存变量：

```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build --preset release -j$(nproc)
```

### 指定 CUDA 路径

如果 CMake 找不到 CUDA，请显式指定路径：

**Linux/macOS：**
```bash
cmake --preset release \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

**Windows (PowerShell)：**
```powershell
cmake --preset release `
      -DCUDAToolkit_ROOT="$env:CUDA_PATH" `
      -DCMAKE_CUDA_COMPILER="$env:CUDA_PATH\bin\nvcc.exe"
```

### 跨平台库扩展名

| 平台 | 共享库扩展名 | 静态库扩展名 |
|------|--------------|--------------|
| Linux | `.so` | `.a` |
| macOS | `.dylib` | `.a` |
| Windows | `.dll` | `.lib` |

---

## 构建选项

| CMake 选项 | 默认值 | 说明 |
|------------|--------|------|
| `BUILD_TESTS` | ON | 构建 GoogleTest 测试套件 |
| `ENABLE_RAPIDCHECK` | OFF | 启用 RapidCheck 基于属性的测试 |
| `BUILD_SHARED_LIBS` | ON | 构建为共享库（`*.so`/`.dll`/`.dylib`） |
| `BUILD_EXAMPLES` | ON | 构建示例程序 |
| `BUILD_BENCHMARKS` | ON | 构建 Google Benchmark 基准测试（`cuflash_attn_bench`） |
| `CMAKE_CUDA_ARCHITECTURES` | `80;86` | 目标 GPU 架构，如 `-DCMAKE_CUDA_ARCHITECTURES=86` |
| `ENABLE_FAST_MATH` | OFF | 启用 `--use_fast_math` 编译器标志 |

### ENABLE_FAST_MATH

启用激进的数学优化，以精度换取速度：

```bash
cmake --preset release-fast-math
cmake --build --preset release-fast-math
```

**效果：**
- 更快的 `expf()` 和 `logf()` 运算
- 数值精度轻微降低
- 对于大多数深度学习训练任务可接受

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

## 运行测试

### CTest（推荐）

```bash
# 使用 preset 运行所有测试
ctest --preset release --output-on-failure

# 运行特定测试
ctest --preset release -R ForwardTest

# 详细输出
ctest --preset release -V
```

### GoogleTest 直接运行

测试二进制位于 `build/<preset>/`：

```bash
# 运行所有测试
./build/release/cuflash_attn_tests

# 运行特定测试套件
./build/release/cuflash_attn_tests --gtest_filter="ForwardTest*"

# 列出所有可用测试
./build/release/cuflash_attn_tests --gtest_list_tests
```

### PyTorch 对比测试

与 PyTorch 参考进行对比验证数值正确性：

```bash
# 首先构建共享库
cmake --preset release

# 运行对比测试
python tests/integration/test_pytorch_comparison.py
```

**库路径解析：**
1. 环境变量 `CUFLASH_LIB`
2. `build/default/` 目录
3. `build/release/` 目录

**自定义库路径：**
```bash
CUFLASH_LIB=/path/to/libcuflash_attn.so python tests/integration/test_pytorch_comparison.py
```

---

## GPU 架构配置

### 默认支持的架构

| 计算能力 | 架构 | 代表 GPU |
|-------------------|--------------|---------------------|
| sm_70 | Volta | V100 |
| sm_75 | Turing | RTX 2080 Ti |
| sm_80 | Ampere | A100 |
| sm_86 | Ampere | RTX 3090 |
| sm_89 | Ada Lovelace | RTX 4090 |
| sm_90 | Hopper | H100 |

### 定位特定架构

```bash
# 单架构（编译更快）
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3090/A100

# 多架构
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES="80;86;89"

# 虚拟架构
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES="80-virtual"  # 虚拟架构
```

### 架构选择指南

| 使用场景 | 推荐配置 | 原因 |
|----------|---------------------------|--------|
| 开发 | `-DCMAKE_CUDA_ARCHITECTURES=86` | 编译更快 |
| A100 集群 | `-DCMAKE_CUDA_ARCHITECTURES=80` | 针对目标优化 |
| H100 集群 | `-DCMAKE_CUDA_ARCHITECTURES=90` | 针对目标优化 |
| 公开发布 | 默认（所有架构） | 最大兼容性 |

### 检查您的 GPU 架构

```bash
# 使用 nvidia-smi（显示当前 GPU）
nvidia-smi

# 使用 deviceQuery（CUDA SDK 示例）
/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# 使用 nvcc
nvcc -arch=sm_70 --run - <<< '__global__ void k(){} int main(){k<<<1,1>>>();}'
```

---

## 跨平台说明

### Linux

标准工作流程开箱即用：

```bash
sudo apt-get install cmake g++  # Ubuntu/Debian
cmake --preset release
cmake --build --preset release -j$(nproc)
```

### macOS

macOS 上的 CUDA 支持仅限于旧版本。NVIDIA 不再为较新的 GPU 提供 macOS 驱动。

### Windows

**使用 Visual Studio 2019/2022：**

```cmd
cmake --preset release
cmake --build --preset release
```

**使用 Ninja（更快）：**

```cmd
cmake --preset release
cmake --build --preset release
```

**常见的 Windows 问题：**
- 确保 CUDA bin 目录在 PATH 中
- 使用 Visual Studio 的 x64 Native Tools Command Prompt

### Docker

```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y cmake g++ git

WORKDIR /workspace
COPY . .

RUN cmake --preset release && \
    cmake --build --preset release
```

运行：
```bash
docker build -t cuflash-attn .
docker run --gpus all cuflash-attn ./build/release/cuflash_attn_tests
```

---

## 故障排除

### CMake 找不到 CUDA

```bash
# 显式设置 CUDA 路径
cmake --preset release \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### 未知的 GPU 架构编译错误

错误：`Unknown CUDA architecture sm_XX`

**解决方案：** 更新 CUDA toolkit 或指定支持的架构：
```bash
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES="70;75;80"
```

### 构建期间内存不足

减少并行任务：
```bash
cmake --build --preset release -j2  # 仅使用 2 个并行任务
```

### 共享库的链接错误

确保 `LD_LIBRARY_PATH` 包含构建目录：
```bash
export LD_LIBRARY_PATH=$PWD/build/release:$LD_LIBRARY_PATH
```

### 无 GPU 系统上的测试失败

CI 工作流配置为在没有 GPU 时仅运行格式检查。要在本地运行测试，请确保您有支持 CUDA 的 GPU。

---

## 下一步

- 阅读 [API 参考](api-reference.md) 了解使用示例
- 查看 [算法文档](algorithm.md) 了解实现细节
- 参考 [故障排除](troubleshooting.md) 了解常见问题
