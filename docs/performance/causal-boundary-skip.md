# Causal 边界块跳过优化（E2b）实测记录

> **数据快照**：2026-08-18（RTX 3060 Laptop 本机实测）
> **硬件**：NVIDIA GeForce RTX 3060 Laptop GPU（`sm_86`，6144 MiB），CUDA 12.0
> **before commit**：`d144765`（E2a，含 grid.y 展平修复）
> **after commit**：本提交（E2b，hash 见 `git log`）
> **计时**：Google Benchmark `UseManualTime()`，CUDA Event 统计 kernel-only 耗时，预热后取中位数。

## 改动内容

forward kernel（`src/forward/flash_attention_forward_typed.cu` 与
`src/forward/flash_attention_forward_wmma.cu`）的 causal KV 块循环在加载
K/V tile 之前，用当前 Q 块最后可见位置做整块跳过判断：

```cuda
const int q_last = min(q_start + BLOCK_M - 1, seq_len - 1);
// causal：整块都在"未来"且后续块必然更远 → 直接结束循环
if (causal && kv_start > q_last) break;
```

- `q_last` 被 clamp 到 `seq_len - 1`，避免最后一块 Q 用未截断的
  `q_start + BLOCK_M - 1` 做比较（对越界 padding 行过度保留）。
- 只允许 `break`（块按 `kv_start` 递增），不允许 `continue` 跳过中间块；
  部分重叠块仍由 `kv <= q_row` 的 mask 原逻辑处理。
- 回归测试：`ForwardTest.CausalNonTileAlignedSeqLen`（`seq_len=257`，
  非整 tile 边界）验证 `q_last` 边界计算正确。

> 说明：D 阶段起 forward kernel 已有 `kv_start > q_start + BLOCK_M - 1`
> 的 break；本优化把比较基准换成 clamp 后的 `q_last`，语义上对合法 KV 块
> 等价，主要价值是显式处理最后一块的越界边界、减少无效访存。

## Before / After（FP32 causal，head_dim=64）

| seq_len | before (ms) | after (ms) | 变化 |
|--------:|------------:|-----------:|-----:|
| 256 | 0.518 | 0.524 | +1.2% |
| 512 | 1.53 | 1.52 | −0.7% |
| 1024 | 4.63 | 4.59 | −0.9% |
| 2048 | 15.9 | 15.6 | −1.9% |
| 4096 | 58.4 | 57.7 | −1.2% |
| 4096 (hd128) | 191 | 189 | −1.0% |

复现命令：

```bash
./build/release/cuflash_attn_bench --benchmark_filter='Forward_Causal'
```

## 非 causal 参考（FP16，head_dim=64）

| seq_len | before (ms) | after (ms) |
|--------:|------------:|-----------:|
| 256 | 0.159 | 0.162 |
| 512 | 0.507 | 0.503 |
| 1024 | 1.82 | 1.80 |
| 2048 | 6.52 | 6.43 |
| 4096 | 24.2 | 24.0 |

复现命令：

```bash
./build/release/cuflash_attn_bench --benchmark_filter='Forward_FP16'
```

## 结论（诚实记录）

- 各长度 causal 耗时变化在 ±2% 内，**增益低于噪声**，未达 10% 阈值。
  原因：causal 整块跳过在 D 阶段已存在，本改动是把比较基准换成 clamp 后的
  `q_last`，对合法 KV 块语义等价。
- **保留改动**（数值不回归，全部 71 项 ctest 全绿）：主要价值是
  **减少无效访存**——对最后一个部分 Q 块显式以 `seq_len - 1` 为界，
  避免对越界 padding 行过度保留 KV 块，使因果边界语义自文档化。
