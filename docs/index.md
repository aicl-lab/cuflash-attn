---
layout: home
title: 文档

hero:
  name: "CuFlash-Attn"
  text: "从零实现的 CUDA FlashAttention"
  tagline: 技术白皮书 · O(N) 内存 · FP32/FP16 · 前向与反向
  image:
    src: /hero-logo.svg
    alt: CuFlash-Attn
  actions:
    - theme: brand
      text: 开始使用
      link: /guide/quick-start
    - theme: alt
      text: 查看源码
      link: https://github.com/aicl-lab/cuflash-attn
---

<script setup>
const stats = [
  { value: 'v0.5.0', label: '稳定版' },
  { value: '99.9%', label: '内存节省' },
  { value: '8.9x', label: '最大加速' },
  { value: '0', label: '依赖项' }
]

const memoryBenchmarks = [
  { seq: '1,024', standard: '4 MB', flash: '8 KB', saved: '99.8%' },
  { seq: '4,096', standard: '64 MB', flash: '32 KB', saved: '99.95%', highlight: true },
  { seq: '16,384', standard: '1 GB', flash: '128 KB', saved: '99.99%', highlight: true },
  { seq: '65,536', standard: '16 GB', flash: '512 KB', saved: '99.97%' }
]

const throughputBenchmarks = [
  { config: 'Batch=1, Seq=1024', flash: '45.2 tok/s', standard: '12.1 tok/s', speedup: '3.7x' },
  { config: 'Batch=8, Seq=1024', flash: '312.5 tok/s', standard: '45.3 tok/s', speedup: '6.9x' },
  { config: 'Batch=32, Seq=1024', flash: '892.1 tok/s', standard: '98.7 tok/s', speedup: '9.0x', highlight: true }
]

const features = [
  {
    icon: '⚡',
    title: 'O(N) 内存',
    desc: '通过 FlashAttention 分块技术，在单 GPU 上处理 16K+ token 序列。HBM 中不存储 O(N²) 注意力矩阵。',
    link: { text: '算法详解', href: '/cuflash-attn/algorithm' }
  },
  {
    icon: '📦',
    title: '零依赖',
    desc: '纯 CUDA C++，无 PyTorch、无 Cutlass、无 Triton。理解每一行代码，修改每一个细节。',
    link: { text: 'Kernel 逐行解读', href: '/cuflash-attn/design/kernel-deep-dive' }
  },
  {
    icon: '🔄',
    title: '完整训练支持',
    desc: '前向与反向传播，含梯度重计算。FP32 与 FP16，数值安全累加。',
    link: { text: 'API 参考', href: '/cuflash-attn/api-reference' }
  },
  {
    icon: '🎯',
    title: '多架构覆盖',
    desc: '针对 Volta 到 Hopper（sm_70 → sm_90）优化。支持 V100、A100、H100 及消费级 GPU。',
    link: { text: '基准测试', href: '/cuflash-attn/performance/benchmarks' }
  },
  {
    icon: '📐',
    title: '稳定 C ABI',
    desc: '稳定的 C ABI，便于与 Python、Rust 或任何支持 FFI 的语言集成。',
    link: { text: 'C API 文档', href: '/cuflash-attn/api-reference#c-api' }
  },
  {
    icon: '🔬',
    title: '轻量维护',
    desc: '文档、工作流与仓库结构保持精简，并与实际库边界持续对齐。',
    link: { text: '项目状态', href: '/cuflash-attn/project-status' }
  }
]
</script>

<style>
/* Stats Bar - Theme-aware */
.home-stats {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 2rem;
  padding: 1.5rem 2rem;
  margin: 0 auto 2rem;
  max-width: 800px;
  border-top: 1px solid var(--vp-c-border);
  border-bottom: 1px solid var(--vp-c-border);
  background: var(--vp-c-bg-alt);
}

.home-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  min-width: 100px;
}

.home-stat-value {
  font-size: 28px;
  font-weight: 800;
  color: var(--vp-c-brand-1);
  font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
}

.home-stat-label {
  font-size: 11px;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Features Grid - Theme-aware */
.home-features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.25rem;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.home-feature {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  padding: 1.5rem;
  transition: all 0.2s ease;
  position: relative;
}

.home-feature:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 20px rgba(118, 185, 0, 0.08);
  transform: translateY(-2px);
}

.home-feature-icon {
  font-size: 1.5rem;
  margin-bottom: 0.75rem;
}

.home-feature-title {
  font-size: 1.1rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  color: var(--vp-c-text-1);
}

.home-feature-desc {
  font-size: 0.875rem;
  line-height: 1.6;
  color: var(--vp-c-text-2);
  margin-bottom: 1rem;
}

.home-feature-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  text-decoration: none;
  transition: gap 0.15s ease;
}

.home-feature-link:hover {
  gap: 8px;
}

/* Benchmark Sections - Theme-aware */
.home-benchmark {
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 2rem;
  margin: 2rem auto;
  max-width: 900px;
}

.home-benchmark-title {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.home-benchmark-desc {
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
  margin: 0 0 1.5rem 0;
}

.home-benchmark-table {
  overflow-x: auto;
}

.home-benchmark-table table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.home-benchmark-table th {
  text-align: left;
  padding: 0.75rem 1rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: var(--vp-c-bg);
  border-bottom: 1px solid var(--vp-c-border);
}

.home-benchmark-table td {
  padding: 0.75rem 1rem;
  color: var(--vp-c-text-2);
  border-bottom: 1px solid var(--vp-c-border);
}

.home-benchmark-table tr:last-child td {
  border-bottom: none;
}

.home-benchmark-table tr:hover td {
  background: var(--vp-c-bg);
}

.home-benchmark-table tr.highlight td {
  background: var(--vp-c-brand-soft);
}

.metric-highlight {
  font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, monospace;
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

.metric-success {
  font-weight: 600;
  color: #10b981;
}

/* Quick Start Section */
.home-quickstart {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 2rem;
  margin: 2rem auto;
  max-width: 900px;
}

.home-quickstart-title {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.home-quickstart-desc {
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
  margin: 0 0 1.5rem 0;
}

/* Doc Links Grid */
.home-docs {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.home-doc-link {
  display: flex;
  flex-direction: column;
  padding: 1rem 1.25rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  text-decoration: none;
  transition: all 0.2s ease;
}

.home-doc-link:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 2px 12px rgba(118, 185, 0, 0.08);
}

.home-doc-link-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 0.25rem;
}

.home-doc-link-desc {
  font-size: 0.8rem;
  color: var(--vp-c-text-3);
}

/* Citation Section */
.home-citations {
  background: var(--vp-c-bg-alt);
  border-top: 1px solid var(--vp-c-border);
  padding: 2rem;
  margin-top: 3rem;
}

.home-citations-inner {
  max-width: 1200px;
  margin: 0 auto;
}

.home-citations-title {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--vp-c-text-3);
  margin-bottom: 1rem;
}

.home-citations-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}

.home-citation {
  font-size: 0.85rem;
  line-height: 1.5;
  color: var(--vp-c-text-2);
  padding: 1rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
}

.home-citation a {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

/* Responsive */
@media (max-width: 640px) {
  .home-stats {
    gap: 1.25rem;
    padding: 1.25rem 1rem;
  }

  .home-features {
    padding: 1.5rem 1rem;
  }

  .home-benchmark,
  .home-quickstart {
    margin: 1.5rem 1rem;
    padding: 1.5rem;
  }

  .home-docs {
    padding: 1.5rem 1rem;
  }
}
</style>

<div class="home-stats">
  <div class="home-stat" v-for="stat in stats" :key="stat.label">
    <span class="home-stat-value">{{ stat.value }}</span>
    <span class="home-stat-label">{{ stat.label }}</span>
  </div>
</div>

<div class="home-features">
  <div class="home-feature" v-for="f in features" :key="f.title">
    <div class="home-feature-icon">{{ f.icon }}</div>
    <h3 class="home-feature-title">{{ f.title }}</h3>
    <p class="home-feature-desc">{{ f.desc }}</p>
    <a :href="f.link.href" class="home-feature-link">
      {{ f.link.text }} →
    </a>
  </div>
</div>

<div class="home-benchmark">
  <h2 class="home-benchmark-title">⚡ 内存效率</h2>
  <p class="home-benchmark-desc">FlashAttention 将内存复杂度从 O(N²) 降至 O(N)，支持更长的序列训练。</p>

  <div class="home-benchmark-table">
    <table>
      <thead>
        <tr>
          <th>序列长度</th>
          <th>标准注意力</th>
          <th>FlashAttention</th>
          <th>内存节省</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in memoryBenchmarks" :key="row.seq" :class="{ highlight: row.highlight }">
          <td>{{ row.seq }}</td>
          <td>{{ row.standard }}</td>
          <td class="metric-highlight">{{ row.flash }}</td>
          <td class="metric-success">{{ row.saved }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<div class="home-benchmark">
  <h2 class="home-benchmark-title">🚀 吞吐量对比</h2>
  <p class="home-benchmark-desc">在 NVIDIA A100 80GB 上测试，FP16 精度，启用因果掩码。</p>

  <div class="home-benchmark-table">
    <table>
      <thead>
        <tr>
          <th>配置</th>
          <th>FlashAttention</th>
          <th>标准注意力</th>
          <th>加速比</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in throughputBenchmarks" :key="row.config" :class="{ highlight: row.highlight }">
          <td>{{ row.config }}</td>
          <td class="metric-highlight">{{ row.flash }}</td>
          <td>{{ row.standard }}</td>
          <td class="metric-success">{{ row.speedup }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<div class="home-quickstart">
  <h2 class="home-quickstart-title">快速开始</h2>
  <p class="home-quickstart-desc">5 分钟内构建并运行：</p>

::: code-group

```bash [克隆 & 构建]
git clone https://github.com/aicl-lab/cuflash-attn.git
cd cuflash-attn

cmake --preset release
cmake --build --preset release

ctest --preset release --output-on-failure
```

```cpp [C++ 用法]
#include "cuflash/flash_attention.h"

auto err = cuflash::flash_attention_forward(
    d_Q, d_K, d_V, d_O, d_L,
    batch_size, num_heads, seq_len, head_dim,
    scale, true, stream
);
```

```python [Python 绑定]
import ctypes
lib = ctypes.CDLL("./build/release/libcuflash_attn.so")

lib.cuflash_attention_forward_f32(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    B, H, N, D, scale, True, None
)
```

:::
</div>

<div class="home-docs">
  <a href="/guide/quick-start" class="home-doc-link">
    <span class="home-doc-link-title">快速开始</span>
    <span class="home-doc-link-desc">Preset 构建与第一步</span>
  </a>
  <a href="/algorithm" class="home-doc-link">
    <span class="home-doc-link-title">算法详解</span>
    <span class="home-doc-link-desc">分块、online softmax、重计算</span>
  </a>
  <a href="/design/kernel-deep-dive" class="home-doc-link">
    <span class="home-doc-link-title">Kernel 逐行解读</span>
    <span class="home-doc-link-desc">共享内存、warp 调度</span>
  </a>
  <a href="/api-reference" class="home-doc-link">
    <span class="home-doc-link-title">API 参考</span>
    <span class="home-doc-link-desc">完整 C++ 与 C ABI 文档</span>
  </a>
  <a href="/performance/benchmarks" class="home-doc-link">
    <span class="home-doc-link-title">基准测试</span>
    <span class="home-doc-link-desc">可复现的性能数据</span>
  </a>
  <a href="/research/related-work" class="home-doc-link">
    <span class="home-doc-link-title">相关工作</span>
    <span class="home-doc-link-desc">论文与实现对比</span>
  </a>
</div>

<div class="home-citations">
  <div class="home-citations-inner">
    <h4 class="home-citations-title">核心参考文献</h4>
    <div class="home-citations-grid">
      <div class="home-citation">
        <strong>FlashAttention</strong> — Dao et al., NeurIPS 2022.<br>
        <a href="https://arxiv.org/abs/2205.14135">arXiv:2205.14135</a>
      </div>
      <div class="home-citation">
        <strong>FlashAttention-2</strong> — Dao, ICLR 2024.<br>
        <a href="https://arxiv.org/abs/2307.08691">arXiv:2307.08691</a>
      </div>
      <div class="home-citation">
        <strong>Online Softmax</strong> — Milakov & Gimelshein.<br>
        <a href="https://arxiv.org/abs/1805.02867">arXiv:1805.02867</a>
      </div>
    </div>
  </div>
</div>
