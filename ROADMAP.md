# CuFlash-Attn 路线图

> 目标：把"从零实现的 FlashAttention"从正确的教学实现，推进为**有优化迭代叙事的 kernel 深度作品**。
> 原则：所有性能数字必须来自当前版本在真实硬件上的测量，并附复现方式。

## 阶段 1：数据刷新（低成本，先做）

- [x] 在可用 GPU 上复测 benchmark 矩阵（RTX 3060 Laptop，commit `6860cbc`，2026-08-18）
- [x] 更新 docs/performance/benchmarks.md 的快照版本与日期（历史 V100/A100/H100 标为 v0.4.0）
- [x] 补齐 head_dim=128 的 benchmark 覆盖

## 阶段 2：一轮有数字的优化迭代

选择 1–2 项，每项记录 before/after 与 profiling 证据（nsys/ncu）：

- [ ] 双缓冲 / 异步拷贝（cp.async）隐藏 HBM 延迟
- [ ] softmax 归约的 warp 级重构（减少 shared memory 往返）
- [x] causal mask 的边界块特化（跳过整块掩码判断）——2026-08-18 E2b，
      FP32 causal 实测约 ±2%，**诚实记录为噪声内负结果**，见
      `docs/performance/causal-boundary-skip.md`
- [x] grid.y > 65535 的 launch 越界修复 + 回归测试（E2a，B*H>65535）

**完成证据**：优化项有独立 before/after 数字。causal 跳过未把 SDPA 比值从
0.42×–0.67× 抬上去，不把负结果改写成加速。

## 阶段 3：面向推理场景的扩展（面试加分项）

- [x] **FlashDecoding / Split-KV**：decode 阶段按 KV 分块并行 + reduce
      （`src/forward/flash_decoding.cu`，数值测试 + benchmark）
- [x] decode 场景（query_len=1）的专项 kernel 与 benchmark
- [ ] （可选）BF16 路径的数值稳定性压测（长序列累积误差）

## 阶段 4：输出与沉淀

- [ ] 写一篇深度文章：FlashAttention 前向/反向推导 + 本实现的优化过程与数字
- [ ] 反向传播的数值稳定性说明文档（当前已有测试，补叙述）

> **面试就绪冻结**（`phase-2-e`）：不再扩新功能。未勾选项（双缓冲、warp softmax、
> BF16 长序列压测、深度文章）记入冻结清单，直到明确解冻。

## 面试讲述要点（完成后自查）

1. 能推导 online softmax，解释为什么不需要物化 O(N²) 矩阵
2. 能解释本实现与 FA2/FA3 的差距来源（warp specialization、TMA 等），以及追赶路径
3. 每个性能数字有硬件、基线、复现方式三要素
