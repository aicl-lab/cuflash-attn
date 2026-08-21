# 项目状态

CuFlash-Attn 当前维护为 **稳定的 v0.5.0 参考实现**，适合学习、审计与轻量集成。

## 保留范围

- 从零实现的 CUDA C++ FlashAttention
- `float`、`half`、`__nv_bfloat16`（BF16）的前向、反向传播
- 前向的 FP16/BF16 在支持架构上走 WMMA（Tensor Core）路径
- 支持的 `head_dim`：`32`、`64`、`128`
- 面向 C++ 与 `ctypes` 的稳定公开接口
- 双语技术文档与 GitHub Pages

## 维护策略

仓库当前优先：

1. **清晰胜过流程堆叠**
2. **删除胜过继续堆框架**
3. **稳定胜过投机扩展**
4. **每类信息只保留一个权威入口**

## 权威入口

- [快速开始](/guide/quick-start)
- [API 参考](/api-reference)
- [系统架构](/architecture)
- [仓库 README](https://github.com/open-infra-ai/cuflash-attn/blob/master/README.md)
- [CHANGELOG.md](https://github.com/open-infra-ai/cuflash-attn/blob/master/CHANGELOG.md)

## 验证边界

- 完整 CUDA 构建依赖可用的 toolkit 与 `nvcc`
- 文档环境不一定具备 GPU 测试条件
- 文档与工作流收敛后，至少应保证仓库结构与文档站可持续构建
