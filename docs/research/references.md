# 参考文献

本页以 BibTeX 风格收录 CuFlash-Attn 设计与实现过程中直接引用的核心文献，按类别编排，便于学术引用与交叉验证。

---

## 目录

- [核心论文](#核心论文)
- [实现参考](#实现参考)
- [性能优化参考](#性能优化参考)
- [CUDA 编程参考](#cuda-编程参考)

---

## 核心论文

以下文献直接定义了 FlashAttention 算法的数学基础、分块策略与数值稳定性机制，是阅读 CuFlash-Attn 源码的必读材料。

---

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**

- **作者**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- **会议**: Advances in Neural Information Processing Systems (NeurIPS), 2022
- **年份**: 2022
- **URL**: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

```bibtex
@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
  url={https://arxiv.org/abs/2205.14135}
}
```

> 本项目关联：前向与反向传播的核心 tiling 算法、SRAM/HBM IO 模型、online softmax 增量更新公式均严格遵循该论文的算法 1–3。

---

**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**

- **作者**: Tri Dao
- **会议**: International Conference on Learning Representations (ICLR), 2024
- **年份**: 2024
- **URL**: [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)

```bibtex
@inproceedings{dao2024flashattention2,
  title={FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations},
  year={2024},
  url={https://arxiv.org/abs/2307.08691}
}
```

> 本项目关联：未来版本中 warpgroup 并行划分与更细粒度 KV 拆分的主要优化方向。

---

**Online normalizer calculation for softmax**

- **作者**: Maxim Milakov, Natalia Gimelshein
- **年份**: 2018
- **URL**: [https://arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867)

```bibtex
@article{milakov2018onlinesoftmax,
  title={Online normalizer calculation for softmax},
  author={Milakov, Maxim and Gimelshein, Natalia},
  journal={arXiv preprint arXiv:1805.02867},
  year={2018},
  url={https://arxiv.org/abs/1805.02867}
}
```

> 本项目关联：Kernel 中 `m_new = max(m_old, rowmax)` 与 `l_new = exp(m_old - m_new) * l_old + rowsum(P)` 的增量更新逻辑直接来源于该工作的流式 softmax 归一化理论。

---

**Multi-Query Attention**

- **作者**: Noam Shazeer
- **年份**: 2019
- **URL**: [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)

```bibtex
@article{shazeer2019mqa,
  title={Fast Transformer Decoding: One Write-Head is All You Need},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:1911.02150},
  year={2019},
  url={https://arxiv.org/abs/1911.02150}
}
```

> 本项目关联：KV Cache 压缩与解码阶段带宽优化的理论基础。本项目当前实现 MHA，但 tile 设计可兼容 MQA/GQA 扩展。

---

**Grouped-Query Attention**

- **作者**: Joshua Ainslie, Tao Lei, Michiel de Jong, Santiago Ontanon, Siddhartha Brahma, Yury Zemlyanskiy, David Uthus, Mandy Guo
- **年份**: 2023
- **URL**: [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

```bibtex
@article{ainslie2023gqa,
  title={{GQA}: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lei, Tao and de Jong, Michiel and Ontanon, Santiago and Brahma, Siddhartha and Zemlyanskiy, Yury and Uthus, David and Guo, Mandy},
  journal={arXiv preprint arXiv:2305.13245},
  year={2023},
  url={https://arxiv.org/abs/2305.13245}
}
```

> 本项目关联：GQA 对 KV 头数量的缩减要求 attention kernel 在 head 维度上具备灵活的 tile 划分能力。

---

**PagedAttention: vLLM**

- **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **会议**: ACM Symposium on Operating Systems Principles (SOSP), 2023
- **年份**: 2023
- **URL**: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

```bibtex
@inproceedings{kwon2023vllm,
  title={Efficient Memory Management for Large Language Model Serving with {PagedAttention}},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion},
  booktitle={ACM Symposium on Operating Systems Principles},
  year={2023},
  url={https://arxiv.org/abs/2309.06180}
}
```

> 本项目关联：PagedAttention 的块稀疏 KV Cache 管理与 FlashAttention 的块计算形成互补；理解该工作是构建端到端推理系统的必要环节。

---

**Ring Attention with Blockwise Transformers for Near-Infinite Context**

- **作者**: Hao Liu, Matei Zaharia, Pieter Abbeel
- **年份**: 2023
- **URL**: [https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)

```bibtex
@article{liu2023ringattention,
  title={Ring Attention with Blockwise Transformers for Near-Infinite Context},
  author={Liu, Hao and Zaharia, Matei and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2310.01889},
  year={2023},
  url={https://arxiv.org/abs/2310.01889}
}
```

> 本项目关联：Ring Attention 将单卡 FlashAttention tiling 扩展到多设备通信场景；本项目作为其底层 kernel 的可审计替代。

---

**Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**

- **作者**: Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia
- **会议**: International Conference for High Performance Computing, Networking, Storage and Analysis (SC), 2021
- **年份**: 2021
- **URL**: [https://arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473)

```bibtex
@inproceedings{narayanan2021megatron,
  title={Efficient Large-Scale Language Model Training on {GPU} Clusters Using {Megatron-LM}},
  author={Narayanan, Deepak and Shoeybi, Mohammad and Casper, Jared and LeGresley, Patrick and Patwary, Mostofa and Korthikanti, Vijay Anand and Vainbrand, Dmitri and Kashinkunti, Prethvi and Bernauer, Julie and Catanzaro, Bryan and Phanishayee, Amar and Zaharia, Matei},
  booktitle={International Conference for High Performance Computing, Networking, Storage and Analysis},
  year={2021},
  url={https://arxiv.org/abs/2104.04473}
}
```

> 本项目关联：为理解 Transformer 分布式训练中的注意力层通信与内存瓶颈提供系统级上下文。

---

## 实现参考

以下仓库与教程为 CuFlash-Attn 的代码结构、API 设计与工程实践提供了直接参考。

---

**Dao-AILab/flash-attention (官方实现)**

- **作者**: Tri Dao 及社区贡献者
- **年份**: 2022–至今
- **URL**: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

```bibtex
@software{flashattention2022github,
  title={FlashAttention},
  author={Dao, Tri and others},
  year={2022},
  url={https://github.com/Dao-AILab/flash-attention},
  note={Official CUDA implementation with PyTorch integration}
}
```

> 本项目关联：算法正确性的主要对标基准；集成测试中的数值等价性验证（误差 < 1e-3）即针对该实现。

---

**NVIDIA CUTLASS (FlashAttention 模板)**

- **作者**: NVIDIA Corporation
- **年份**: 2017–至今
- **URL**: [https://github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

```bibtex
@software{cutlass2022github,
  title={{CUTLASS}: {CUDA} Templates for Linear Algebra Subroutines and Solvers},
  author={{NVIDIA Corporation}},
  year={2022},
  url={https://github.com/NVIDIA/cutlass},
  note={Version 3.x includes FlashAttention kernel templates}
}
```

> 本项目关联：对比理解模板元编程与显式 CUDA kernel 两种实现路径的工程权衡。

---

**OpenAI Triton (FlashAttention Tutorial)**

- **作者**: Philippe Tillet 及 Triton 社区
- **年份**: 2021–至今
- **URL**: [https://github.com/openai/triton](https://github.com/openai/triton)

```bibtex
@software{triton2021github,
  title={Triton: Language for {GPU} Kernel Development},
  author={Tillet, Philippe and others},
  year={2021},
  url={https://github.com/openai/triton},
  note={Includes Python-level FlashAttention tutorial implementation}
}
```

> 本项目关联：Triton tutorial 提供了高层次的 kernel 设计思路，CuFlash-Attn 将其映射为显式 CUDA C++ 实现，以暴露底层硬件执行细节。

---

## 性能优化参考

以下文献为 GPU kernel 性能分析、Roofline 建模与内存优化提供了方法论支撑。

---

**Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures**

- **作者**: Samuel Williams, Andrew Waterman, David Patterson
- **会议**: Communications of the ACM, 2009
- **年份**: 2009
- **URL**: [https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/Roofline.pdf](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/Roofline.pdf)

```bibtex
@article{williams2009roofline,
  title={Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures},
  author={Williams, Samuel and Waterman, Andrew and Patterson, David},
  journal={Communications of the ACM},
  volume={52},
  number={4},
  pages={65--76},
  year={2009},
  url={https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/Roofline.pdf}
}
```

> 本项目关联：本项目性能分析页面中的 Roofline 图表直接应用该模型，用于判定 kernel 处于带宽瓶颈还是计算瓶颈。

---

**Dissecting the NVIDIA Ampere Architecture through Microbenchmarking and Instruction-level Analysis**

- **作者**: Zhe Jia, Marco Maggioni, Benjamin Staiger, Daniele Paolo Scarpazza
- **会议**: IEEE International Parallel and Distributed Processing Symposium (IPDPSW), 2022
- **年份**: 2022
- **URL**: [https://arxiv.org/abs/2208.11164](https://arxiv.org/abs/2208.11164)

```bibtex
@inproceedings{jia2022ampere,
  title={Dissecting the {NVIDIA} {Ampere} Architecture through Microbenchmarking and Instruction-level Analysis},
  author={Jia, Zhe and Maggioni, Marco and Staiger, Benjamin and Scarpazza, Daniele Paolo},
  booktitle={IEEE International Parallel and Distributed Processing Symposium Workshops},
  year={2022},
  url={https://arxiv.org/abs/2208.11164}
}
```

> 本项目关联：Ampere (A100) 与 Hopper (H100) 架构的共享内存带宽、Tensor Core 行为与 warp 调度细节的重要微观基准参考。

---

## CUDA 编程参考

以下 NVIDIA 官方文档是 CUDA kernel 开发的事实标准参考。

---

**CUDA C++ Programming Guide**

- **作者**: NVIDIA Corporation
- **年份**: 2024（持续更新）
- **URL**: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

```bibtex
@manual{nvidia2024cudaguide,
  title={{CUDA C++} Programming Guide},
  author={{NVIDIA Corporation}},
  year={2024},
  url={https://docs.nvidia.com/cuda/cuda-c-programming-guide/},
  note={Version 12.x}
}
```

> 本项目关联：共享内存组织、`__launch_bounds__`、warp 级原语、异步内存拷贝等 CUDA 特性的权威文档来源。

---

**NVIDIA CUDA Best Practices Guide**

- **作者**: NVIDIA Corporation
- **年份**: 2024（持续更新）
- **URL**: [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

```bibtex
@manual{nvidia2024cudabestpractices,
  title={{NVIDIA CUDA} Best Practices Guide},
  author={{NVIDIA Corporation}},
  year={2024},
  url={https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/},
  note={Coalesced access, occupancy, and shared memory optimization guidelines}
}
```

> 本项目关联：向量化加载（`float4`）、共享内存 bank conflict 避免、 Occupancy 优化的直接参考。

---

**CUDA Binary Utilities**

- **作者**: NVIDIA Corporation
- **年份**: 2024（持续更新）
- **URL**: [https://docs.nvidia.com/cuda/cuda-binary-utilities/](https://docs.nvidia.com/cuda/cuda-binary-utilities/)

```bibtex
@manual{nvidia2024cudabinutils,
  title={{CUDA} Binary Utilities},
  author={{NVIDIA Corporation}},
  year={2024},
  url={https://docs.nvidia.com/cuda/cuda-binary-utilities/},
  note={SASS instruction reference for sm_70 through sm_90}
}
```

> 本项目关联：需要深入分析编译器生成的 SASS 代码、验证 warp 级调度与指令发射模式时的底层参考。

---

## 引用 CuFlash-Attn

如需在学术工作中引用 CuFlash-Attn 本项目，建议使用以下格式：

```bibtex
@software{cuflashattn2024,
  title={CuFlash-Attn: From-Scratch {CUDA} {C++} {FlashAttention} Reference Library},
  author={{AICL-Lab}},
  year={2024},
  url={https://github.com/aicl-lab/cuflash-attn},
  note={Version 0.5.0, stable baseline}
}
```
