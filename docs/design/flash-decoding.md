# FlashDecoding（Split-KV）decode 加速

> 状态：已实现（commit `52c4bfd+` 之后），有差分测试与实测 benchmark。

## 1. 动机

decode 阶段每步只有 **1 个 query**，却要对着整条 KV cache 做 attention
（QK^T + PV）。序列越长，这个 KV 扫描越占时间，且经典 forward kernel 里
它是一条串行路径——一个 block 扫完整个 KV。FlashDecoding（Split-KV）把
KV 沿序列维拆成 `num_chunks` 块，每块一个 block **并行** 计算局部
online-softmax 部分量，最后跨块归约，从而把 decode 的 KV 扫描并行化。

## 2. API

```cpp
cuflash::flash_attention_decode(Q, K, V, O, L,
                                batch_size, num_heads, seq_len, head_dim,
                                scale, num_chunks, stream);
```

- `Q`：`[batch_size * num_heads, 1, head_dim]`
- `K/V`：`[batch_size * num_heads, seq_len, head_dim]`
- `O`：`[batch_size * num_heads, 1, head_dim]`
- `L`：`[batch_size * num_heads]`（logsumexp，FP32）
- `num_chunks`：KV 拆分块数，会被 clamp 到 `ceil(seq_len / 64)` 以内
- 支持 FP32 / FP16 / BF16（`impl::TypeAdapter` 统一写回）

## 3. 两阶段实现

### Phase 1：`flash_decoding_partial_kernel`
grid `(num_chunks, batch_size*num_heads)`，每个 block 处理一个
`(chunk, batch*head)`：

1. 把 query 行载入共享内存；
2. 在其 chunk 范围内按 `BLOCK_N=64` 分 tile 扫描：
   - `scores = Q·K^T * scale`（线程按 key 位置并行，block 归约取 max）；
   - 在线 softmax 更新：`m_new = max(m, tile_max)`，`l = l*rescale + sum_exp`，
     `O_acc = O_acc*rescale + P·V`；
3. 写出局部部分量 `(m_c, l_c, O_c)` 到 scratch。

### Phase 2：`flash_decoding_combine_kernel`
grid `(batch_size*num_heads)`，把各 chunk 的部分量按 FlashAttention 重缩放
规则归约：

```
m = max_c m_c
l = Σ_c l_c · exp(m_c - m)
O = Σ_c O_c · exp(m_c - m) / l
L[bh] = m + log(l)
```

跨块归约用全局 scratch（`[num_chunks, bh_total]` 布局，函数级 static
缓冲复用，避免 decode 每步 `cudaMalloc`）。

## 4. 数值正确性

- 与 CPU 参考（单 query 全 KV 注意力）逐元素比较：FP32 `1e-3`、FP16 `1e-2`；
- **chunk 数不变性**：`num_chunks=1`（= 标准 decode）与 `num_chunks=16`
  输出一致（`1e-4`），证明 Split-KV 归约正确；
- 非法参数（空指针 / 非法 head_dim / 非法维度）返回错误码不崩溃。

测试：`tests/unit/test_flash_decoding.cu`（4 例）。

## 5. 实测（RTX 3060 Laptop，FP16，batch=1，heads=8，head_dim=64）

| KV seq_len | num_chunks | decode 延迟 (ms) |
|:----------:|:----------:|:----------------:|
| 1,024      | 8          | 0.050            |
| 4,096      | 32         | 0.107            |
| 16,384     | 128        | 0.335            |
| 65,536     | 512        | 1.50             |

延迟随 `seq_len` 近线性增长（decode 是带宽受限的 KV 扫描），且被
Split-KV 的并行度摊薄。复现：

```bash
cmake --preset release && cmake --build --preset release -j
./build/release/cuflash_attn_bench --benchmark_filter="BM_Decode_FP16"
```

## 6. 边界与后续

- 当前是两 kernel（partial + combine），未做单 kernel 原子归约；
- 未做 KV 物理块页式布局（本仓库 Q/K/V 是连续 `[bh, S, D]`），
  页式 PagedAttention 控制面在 paged-infer；
- 后续可加 `num_chunks` autotune 或与 FlashAttention-3 式 split 归约合并。
