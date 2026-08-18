# CuFlash-Attn v1.0 开发计划（Phase A/C/D，Phase B 跳过）

> 状态：v1.0 计划草稿（归档），尚未执行；后续 cuflash 优化任务以组织级 PHASE2_PLAN.md 的 E2 为准。

> 本计划由 2026-08 的仓库审查结论生成，供后续低成本模型逐步执行。
> 目标：把当前“正确但文档/性能口径混乱”的教学实现，收敛为一个可信的 v1.0 参考实现和面试作品。

---

## 0. 执行前提与硬性规则

1. **Phase B（decode / KV cache / split-KV / GQA / serving）不在本仓库实施。**
   本计划只做 Phase A（收敛修复）、可选 Phase C（一轮有数字的优化）、Phase D（文章与发布）。
2. 不要追求追平 FlashAttention-2/3，不要引入 CUTLASS、TMA/WGMMA、backward WMMA。
3. 每个任务完成后必须执行：
   ```bash
   cd /home/shane/github/aicl/cuflash-attn
   cmake --preset release
   cmake --build --preset release -j
   ctest --preset release --output-on-failure
   ```
   出现失败时，先修复再进入下一任务。
4. 一次只提交一个任务：
   - 修复类：`fix: ...`
   - 文档类：`docs: ...`
   - 测试类：`test: ...`
   - 发布类：`chore: ...`
5. 没有实测数据时，禁止填写或修改性能数字。禁止把估算值伪装成实测值。
6. 如果执行环境缺少 PyTorch 或 compute-sanitizer，不要伪造“通过”。在对应文档中记录 `BLOCKED: 缺少 xxx`，并保留已有替代证据（ctest、numpy 参考对比）。
7. 除任务明确说明外，不要修改公开 API 签名；如果确需修改，必须同步更新
   `include/cuflash/flash_attention.h`、`docs/api-reference.md`、`CHANGELOG.md` 和 Python/ctypes 示例。

### 当前基线（执行时先确认）

- 基线 commit：`f588314`
- 本机硬件：NVIDIA GeForce RTX 3060（sm_86），CUDA 12.0，CMake 3.28
- 现有测试：`ctest` 65/65 通过（PyTorch 对比被 skip）
- 已知问题来源：仓库审查发现 benchmark 指标错误、文档与代码不一致、HBM 复杂度表述错误、grid.y 65535 限制等

---

## 1. 任务总览

| 编号 | 任务 | 优先级 | 状态 |
|---|---|---|---|
| T1 | 修复 benchmark 计时与性能指标 | P0，必须 | [ ] |
| T2 | 修正文档中的算法复杂度、dtype、路径和定位表述 | P0，必须 | [ ] |
| T3 | 补充非整除 N / dtype / 边界回归测试 | P0，必须 | [ ] |
| T4 | 修复 grid.y 65535 限制与 int 溢出 | P1，必须 | [ ] |
| T5 | 真实 GPU 验证：PyTorch 对比、benchmark 复测、sanitizer 尝试 | P0，必须 | [ ] |
| T6 | 发布 v1.0：CHANGELOG、ROADMAP、技术文章、tag | P0，必须 | [ ] |
| C1 | 可选：一轮有 profiling 数字的性能优化 | P2，可选 | [ ] |

依赖顺序：T1 → T2 → T3 → T4 → T5 → T6。C1 只能在 T5 之后做。

---

## 2. T1：修复 benchmark 计时与性能指标

### 2.1 问题

文件：`benchmarks/bench_flash_attention.cu`

1. `benchmark::Counter(...) / 1e12` 会把 `kIsIterationInvariantRate` 标志丢掉，
   当前打印的 `TFLOP/s` 只是 `FLOPs / 1e12`，不是速率。
2. HBM 字节模型只算 Q/K/V/O 各一次，漏掉 K/V 的 `ceil(N / BLOCK_M)` 次重载。
3. backward FLOPs 用“前向 × 2.5”，与代码实际不符：
   - dQ kernel：QKᵀ、dO·Vᵀ、dS·K，3 次 tile matmul；
   - dKdV kernel：QKᵀ、Pᵀ·dO、dO·Vᵀ、dSᵀ·Q，4 次 tile matmul；
   - 合计 7 次，约 `14 * B*H*N²*D` FLOPs（causal 约一半）。
4. 文档声称 CUDA Event kernel-only 计时，但代码实际用 `cudaStreamSynchronize`
   包住整轮循环，没有 event，也没有 `UseManualTime`。

### 2.2 修改步骤

#### Step 1：修正 FLOPs 模型

在 `bench_flash_attention.cu` 中把 `report_metrics` 重写为：

```cpp
// FLOPs 模型
// forward: 2*N^2*D (QK^T) + 2*N^2*D (PV) = 4*B*H*N^2*D
// causal forward 约减半。
// backward（本实现实际标量路径）：
//   dQ kernel 3 次 tile matmul + dKdV kernel 4 次 tile matmul
//   = 14 * B*H*N^2*D FLOPs；causal 约减半。
static double attention_flops(int batch_size, int num_heads, int seq_len,
                              int head_dim, bool backward, bool causal) {
    const double bh = static_cast<double>(batch_size) * num_heads;
    const double n = static_cast<double>(seq_len);
    const double d = static_cast<double>(head_dim);
    double flops = (backward ? 14.0 : 4.0) * bh * n * n * d;
    if (causal) {
        flops *= 0.5;
    }
    return flops;
}
```

同时把 `report_metrics` 签名改为带 `causal`：

```cpp
static void report_metrics(benchmark::State& state, int batch_size, int num_heads,
                           int seq_len, int head_dim, size_t elem_size,
                           bool backward, bool causal);
```

所有调用点（包括 `BM_Forward_Causal`）同步传入实际 causal 值。

#### Step 2：修正 Counter 用法

不要写 `Counter(...) / 1e9`，直接构造时把值换算好，保留 flags：

```cpp
state.counters["GFLOPS/s"] = benchmark::Counter(
    flops / 1e9,
    benchmark::Counter::kIsIterationInvariantRate,
    benchmark::Counter::OneK::kIs1000);
```

#### Step 3：改为 CUDA Event 计时

给所有 FlashAttention benchmark 函数增加：

```cpp
cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
cudaEventCreate(&ev_start);
cudaEventCreate(&ev_stop);

for (auto _ : state) {
    cudaEventRecord(ev_start, stream);
    auto err = cuflash::flash_attention_forward(...);
    cudaEventRecord(ev_stop, stream);

    if (err != cuflash::FlashAttentionError::SUCCESS) {
        state.SkipWithError("flash_attention_forward failed");
        break;
    }

    cudaEventSynchronize(ev_stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    state.SetIterationTime(elapsed_ms / 1000.0);
}
```

并给对应 `BENCHMARK(...)` 注册添加：

```cpp
->Unit(benchmark::kMillisecond)->UseManualTime();
```

注意事项：
- naive baseline 也可以保留现有 wall-clock 方式，但必须明确它不是生产基线；
- 不要在 benchmark 计时循环里重新 `cudaMalloc`/`cudaFree` 或初始化数据。

#### Step 4：前向 logical HBM 字节模型

只对前向报告 logical HBM，反向 HBM 由 ncu 实测，避免模型继续出错。

给 benchmark target 增加私有 include 目录，`CMakeLists.txt` 中：

```cmake
target_include_directories(cuflash_attn_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src
)
```

在 benchmark 文件中 include：

```cpp
#include <algorithm>
#include "impl/tile_io.cuh"
#include "kernel_launch_utils.cuh"
```

用 `impl::ForwardTilingConfig` 与 `cuflash::query_max_dynamic_shared_memory_per_block`
复现 launcher 的 tile 选择：

```cpp
struct FwdTile { int bm = 0; int bn = 0; };

FwdTile fwd_scalar_tile(int head_dim, int max_dynamic_smem) {
    using C = impl::ForwardTilingConfig;
    if (head_dim == 32) return {C::BLOCK_M, C::BLOCK_N};
    if (head_dim == 64) {
        if (C::smem_bytes(64, C::BLOCK_M, C::BLOCK_N) >
            static_cast<size_t>(max_dynamic_smem)) {
            return {C::BLOCK_M_SMALL, C::BLOCK_N_SMALL};
        }
        return {C::BLOCK_M, C::BLOCK_N};
    }
    // head_dim == 128
    if (C::smem_bytes(128, C::BLOCK_M_HD128, C::BLOCK_N_HD128) >
        static_cast<size_t>(max_dynamic_smem)) {
        return {C::BLOCK_M_HD128_SMALL, C::BLOCK_N_HD128_SMALL};
    }
    return {C::BLOCK_M_HD128, C::BLOCK_N_HD128};
}

FwdTile fwd_wmma_tile(int head_dim) {
    return (head_dim == 128) ? FwdTile{32, 32} : FwdTile{64, 32};
}
```

logical HBM 公式（元素数 × dtype 字节数）：

```cpp
// Q 读 1 次 + O 写 1 次；每个 Q block 要读 K/V 各一遍。
// 非 causal：每个 Q block 读满 N 行 KV。
// causal：Q block 的 q_start+q_len 之前才有效。
// 以完整有效行数近似，不计算边界 tile 的 padding；逻辑字节再乘 dtype size。
double fwd_logical_hbm_elements(int bh, int n, int d, bool causal, int bm) {
    const int q_blocks = (n + bm - 1) / bm;
    double kv_rows = 0.0;
    for (int qb = 0; qb < q_blocks; ++qb) {
        const int q_start = qb * bm;
        const int valid_rows = causal
            ? std::min(q_start + bm, n)
            : n;
        kv_rows += static_cast<double>(valid_rows);
    }
    // Q + O 各 1 次，K/V 各按每个 Q block 的重载次数计
    return 2.0 * static_cast<double>(bh) * n * d +
           2.0 * static_cast<double>(bh) * d * kv_rows;
}
```

- 对 reduced precision benchmark，若当前设备会走 WMMA（half: sm70+，bf16: sm80+），
  用 `fwd_wmma_tile`；否则用 `fwd_scalar_tile`。
- `report_metrics` 中只在 `!backward` 时报告该值：

```cpp
if (!backward) {
    const double logical_bytes =
        fwd_logical_hbm_elements(batch_size * num_heads, seq_len, head_dim,
                                 causal, tile.bm) *
        static_cast<double>(elem_size);
    state.counters["LogicalHBM GB/s"] = benchmark::Counter(
        logical_bytes / 1e9,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
}
```

- counter 名称必须使用 `LogicalHBM GB/s`，不要继续叫裸 `HBM GB/s`。
- 反向 benchmark 不再输出 HBM 计数器，只输出 GFLOPS/s。

#### Step 5：naive baseline 与 causal benchmark 同步修正

- naive baseline 的 FLOPs counter 使用 Step 1/2 的写法；
- naive 的 HBM 模型保留原“S/P 各读一次写一次”公式，这本来就是对比目的；
- `BM_Forward_Causal` 使用 `attention_flops(..., causal=true)`。

### 2.3 T1 验收标准

- [ ] `GFLOPS/s` 打印值与人工计算一致。
  人工校验示例：B=1,H=8,N=1024,D=64 前向 FLOPs=2.147e9；
  若 event 时间 8.1ms，则打印值应约为 `265 GFLOPS/s`（0.265 TFLOPS）。
- [ ] FP16 前向 benchmark 的 `LogicalHBM GB/s` 显著高于旧值（因为补上了 K/V 重载）。
- [ ] `BM_Backward_*` 不再使用 2.5× 模型。
- [ ] 运行：
  ```bash
  ./build/release/cuflash_attn_bench \
    --benchmark_filter='BM_Forward_FP32/1024/64|BM_Forward_FP16/1024/64|BM_Backward_FP32/1024/64'
  ```
  输出时间与事件计时一致，无 `-nan` 或异常值。

---

## 3. T2：修正文档中的错误表述与矛盾

### 3.1 必须修正的核心概念

**HBM 流量复杂度**：当前实现是 FA1 结构（一个 block 串行遍历 KV），HBM 流量为

```
Q 读 1 次 + O 写 1 次 + (K+V) × ceil(N / BLOCK_M)
= Θ(N²·D / BLOCK_M)，BLOCK_M 固定为 64/32
```

因此：
- **工作内存/中间激活是 O(N)**；
- **HBM 流量仍是 O(N²)**，只是常数小于 materialized attention；
- 论文中的 `Θ(N²D²/M)` 需要 block size 随 SRAM 容量合理放大，当前实现没有达到；
- FA2 的 split-KV 才进一步减少 K/V 重载，本仓库没有实现。

修改文件与位置：

1. `docs/algorithm.md`
   - “内存复杂度分析”表格中，HBM IO 列改为：
     - 标准 Attention：`O(N² + Nd)`
     - 本实现：`Θ(N²D/BLOCK_M + ND)`（固定 tile 下即 `O(N²)`）
   - 删掉或改写“当 M=Θ(Nd) 时 IO 趋近 O(Nd)”这种对当前代码不适用的表述。
   - FP16 实现一节更新：v0.5.0 起 FP16/BF16 前向走 WMMA，
     Q/K/V tile 保持输入精度，MMA 累加为 FP32，softmax 后将 P 量化为输入精度做 PV；
     scalar fallback 仍为“load 转 FP32、CUDA core 计算”。

2. `docs/performance/roofline-analysis.md`
   - 所有“FlashAttention HBM 流量 = O(N)”或“仅 Q/K/V/O 各一次”的推导，
     改为按上述公式计算。
   - 4.1 的示例表重新计算；例如 N=16384,D=64,BH=8,FP16，
     K/V 按每个 Q block 重载后是数 GB 级，不是 520 MB。
   - 6.2 对比表中，本实现 HBM scaling 写 `O(N²)`（固定 tile），
     FlashAttention-2/3 写 `O(N)` 或 `split-KV 后大幅降低`，不要混淆。

### 3.2 README 修正

文件：`README.md`

- 标题/简介中的“高性能”改为“教学/参考实现”：
  > 从零实现的 CUDA C++ FlashAttention 教学/参考实现；FP16/BF16 前向已接入 WMMA。
- “支持的参数”数据类型一行加上 BF16：`FP32 (float), FP16 (half), BF16 (__nv_bfloat16)`。
- GPU 架构支持改为：
  > 可配置编译 sm_70–sm_90；默认构建 sm_80/sm_86；仓库当前实测硬件为 sm_86。
- 环境要求统一为：CUDA 11.8+（推荐 12.x；sm_90 需 11.8+；CUDA 13.x 已移除 sm_70）。
- 性能一节保留“请在目标硬件运行 benchmark”，并注明 benchmark 指标口径为
  CUDA Event 计时 + 模型化 logical HBM。

### 3.3 其他文档修正

| 文件 | 修改 |
|---|---|
| `docs/project-status.md` | “保留范围”加上 BF16；去掉与 README 矛盾的表述 |
| `docs/api-reference.md` | 补充 BF16 前向/反向签名，或明确链接到 `flash_attention.h` 为唯一权威 |
| `docs/architecture.md` | 删除不存在的 `flash_attention_c.h`、`forward_kernel_f32.cu` 等；目录树改为真实文件；block 大小改为实际 64/64、32/32（backward 32/32、16/32 等） |
| `docs/building.md` | 修正测试二进制路径（如 `build/release/cuflash_attn_tests`）、PyTorch 测试路径（`tests/integration/test_pytorch_comparison.py`）、benchmark CMake 变量（`BUILD_BENCHMARKS`、`CMAKE_CUDA_ARCHITECTURES`） |
| `docs/guide/quick-start.md` | CUDA 最低版本与 README/CHANGELOG 对齐；修正测试路径 |
| `docs/design/design-decisions.md` | ADR-2 改为“低精度输入、FP32 accumulate；WMMA 前向的 tile 保持输入精度” |
| `docs/design/kernel-deep-dive.md` | 增加 WMMA 路径说明，或明确该文档只描述 scalar 路径 |
| `docs/design/tensor-core-migration.md` | 更新状态：Phase 2 forward 已实现；如 T5 验证完成，把“待真实硬件验证”改为实测结论；backward WMMA 明确“不纳入 v1.0” |
| `docs/performance/benchmarks.md` | 修正复现命令；快照版本、设备、CUDA、日期改为 T5 实测数据 |

### 3.4 T2 验收标准

- [ ] 以下 grep 无旧路径/变量残留（允许 CHANGELOG 历史记录）：
  ```bash
  grep -R "forward_kernel_f32.cu\|flash_attention_c.h" docs README.md || true
  grep -R "CUFASH_ATTN_BENCHMARKS\|CUFASH_ATTN_ARCHS" docs || true
  grep -R "build/release/tests" docs || true
  grep -R "tests/test_pytorch_comparison.py" docs || true
  ```
- [ ] README 中不再同时出现“支持 BF16”和“参数表只有 FP32/FP16”的矛盾。
- [ ] docs 站构建成功：
  ```bash
  cd docs && npm ci && npm run docs:build
  ```
- [ ] 搜索“HBM.*O(N)”等表述时，文档能区分“显存占用 O(N)”与“HBM 流量 O(N²)”。

---

## 4. T3：补充回归测试

### 4.1 新增测试内容

新建 `tests/unit/test_non_divisible_shapes.cu`，加入 CMake 测试源文件列表。

覆盖矩阵（前向与反向都测）：

| dtype | N | D | causal |
|---|---|---|---|
| FP32 | 1, 17, 33, 65, 100 | 32, 64, 128 | false/true |
| FP16 | 1, 17, 33, 65 | 32, 64, 128 | false/true |
| BF16 | 1, 17, 33, 65 | 32, 64, 128 | false/true |

说明：
- N=17/33/65/100 用于覆盖 `seq_len` 不是 block 整数倍且跨多个 KV block 的路径。
- 反向测试先调 forward 得到 `O` 和 `L`，再调 backward，与现有 `test_backward.cu` 一致。
- FP16/BF16 参考实现必须使用“量化后的输入”在 FP32 中计算，再与 kernel 输出比较，
  不要用未量化的 FP32 输入作为参考。
- 容差沿用现有约定：FP32 `2e-3`，FP16/BF16 `2e-2`；如果 BF16 在 D=128 边缘案例
  超出，允许把 BF16 放宽到 `3e-2`，但必须在测试注释中写明原因。
- 额外补一个 `seq_len=1` 的反向测试（三 dtype、causal true/false）。

实现建议：
- 优先把 CPU reference 抽取到 `tests/unit/test_reference.h`，让新旧测试共用；
- 如果抽取成本高，允许在新测试文件中复制 reference 实现，但顶部注明
  “Copy from test_forward.cu / test_backward.cu, kept local to limit refactor risk”。

### 4.2 T3 验收标准

- [ ] 新测试出现在 `ctest` 列表中并全部通过。
- [ ] 非整除 N 的 forward/backward 在三 dtype 下通过。
- [ ] 运行：
  ```bash
  ./build/release/cuflash_attn_tests --gtest_filter='*NonDivisible*:*SeqLenOne*'
  ```
  全部绿色。

---

## 5. T4：修复 grid.y 65535 限制与 int 溢出

### 5.1 问题

`src/forward/flash_attention_forward_typed.cu`、`flash_attention_forward_wmma.cu`、
`src/backward/flash_attention_backward_typed.cu` 都把 `batch_size * num_heads`
放进 `grid.y`（上限 65535），且大量指针偏移用 `int` 乘法。

### 5.2 修改方案

统一改为“一维 grid + q_blocks 分解”。先取一个安全上限：

```cpp
const int64_t batch_heads = static_cast<int64_t>(batch_size) * num_heads;
const int64_t q_blocks = (static_cast<int64_t>(seq_len) + BM - 1) / BM;
const int64_t total_blocks = batch_heads * q_blocks;
constexpr int64_t kMaxGridX = 0x7fffffffLL;

if (total_blocks > kMaxGridX) {
    return FlashAttentionError::INVALID_DIMENSION;  // 实际规模不可达，防御性
}
dim3 grid(static_cast<unsigned>(total_blocks));
```

kernel 开头：

```cpp
const int q_blocks = (seq_len + BM - 1) / BM;
const int64_t linear = static_cast<int64_t>(blockIdx.x);
const int q_block_idx = static_cast<int>(linear % q_blocks);
const int64_t batch_head_idx = linear / q_blocks;
```

- 指针偏移统一先转 `int64_t`：
  ```cpp
  const int64_t bh = batch_head_idx;
  const InputT* Q_ptr = Q + bh * static_cast<int64_t>(seq_len) * HEAD_DIM;
  ```
- 同样修改：
  - `flash_attention_forward_wmma_kernel`
  - `flash_attention_backward_dq_kernel`
  - `flash_attention_backward_dkdv_kernel`
  - `compute_D_kernel`（`compute_D` 的 grid 为 `d_blocks * batch_heads`，
    kernel 内用 `linear % seq_len` 得到 row、`linear / seq_len` 得到 batch-head）
- 反向 launcher 中 `d_grid`、`dq_grid`、`dkdv_grid` 也用同一模式。
- `AsyncFloatBuffer::alloc` 已经用 `size_t`，无需大改。
- 修改 kernel 显式实例化的签名（如果 launcher 的 grid 计算有变化，模板实例化通常不变）。

### 5.3 测试

在 `test_stress_edge_cases.cu` 或新文件增加一个低内存大 batch 冒烟测试：

```cpp
// B*H = 70000 > gridDim.y 上限 65535，用最小张量验证 launch 不再失败
batch_size = 70000, num_heads = 1, seq_len = 1, head_dim = 32, FP32 forward
```

只断言返回 SUCCESS、同步后 CUDA 无错误、输出有限；不比较全量数值。
如果 6GB 显存不足，把 B 降到 66000，但必须仍大于 65535。

### 5.4 T4 验收标准

- [ ] 大 batch 冒烟测试通过。
- [ ] 现有 65 个测试不回归。
- [ ] 用 `grep -n "blockIdx.y\|grid.y" src/` 检查没有遗漏的 batch_heads 路径。

---

## 6. T5：真实 GPU 验证与 benchmark 复测

### 6.1 PyTorch 对比

```bash
python3 -m pip install -r tests/integration/requirements.txt
python3 tests/integration/test_pytorch_comparison.py
```

- 该脚本自带 FP32/FP16/BF16 前向、反向、causal 对比。
- 如果安装 torch 后脚本失败，必须修复脚本或代码，不得改松断言来“通过”。
- 执行成功后，把输出摘要和 `torch.__version__`、GPU 名称记入验证文档。

### 6.2 Benchmark 复测

```bash
cmake --preset release
cmake --build --preset release -j

./build/release/cuflash_attn_bench \
  --benchmark_time_unit=ms \
  --benchmark_min_time=0.2s \
  --benchmark_out=benchmark_v1.0.json \
  --benchmark_out_format=json \
  --benchmark_filter='BM_Forward_FP32|BM_Forward_FP16|BM_Forward_BF16|BM_Backward_FP32|BM_Backward_FP16|BM_Forward_Causal'
```

记录并提交：
- GPU 型号、driver、`nvcc --version`、commit id；
- 更新 `docs/performance/benchmarks.md` 为 v1.0 快照；
- 只写实际跑过的形状，跑不动的标 `未测`。

### 6.3 compute-sanitizer 尝试

```bash
./scripts/run_compute_sanitizer.sh build/release
```

- 如果本机因缺 `libsanitizer-collection.so` 或权限失败，如实记录；
- 不阻塞 v1.0，但必须在验证文档中写明“sanitizer 未执行/失败原因”；
- 如果条件允许，使用 CI 的 GPU workflow 或带完整 CUDA toolkit 的 Docker 环境补跑。

### 6.4 创建验证文档

新建 `docs/performance/validation-v1.0.md`，内容包含：

- 测试设备、驱动、CUDA、torch 版本；
- `ctest` 结果（总测试数 / 通过 / skip）；
- PyTorch 对比摘要；
- benchmark 关键形状表格（时间、GFLOPS/s、LogicalHBM GB/s）；
- sanitizer 执行状态（通过 / 失败原因 / 未执行）；
- 已知限制：FP32 前向和全部反向仍为 scalar、WMMA 仅前向、无 split-KV。

### 6.5 T5 验收标准

- [ ] PyTorch 对比测试无 skip 地通过（至少本机 GPU）。
- [ ] benchmark JSON 已生成，`docs/performance/benchmarks.md` 版本、日期、设备与 JSON 一致。
- [ ] `docs/performance/validation-v1.0.md` 存在且无虚假“通过”。

---

## 7. T6：发布 v1.0 与沉淀

### 7.1 技术文章

新建 `docs/articles/from-scalar-to-wmma.md`（或同级路径），建议结构：

1. 标准 attention 的内存问题；
2. online softmax 推导；
3. 本实现的前向/反向 kernel 结构与 tiling；
4. **关键纠正**：固定 tile 下 HBM 流量仍是 O(N²)，O(N) 只是显存占用；
5. WMMA 前向接入过程与实测加速；
6. 与 FA2/FA3 的差距：split-KV、cp.async、warp specialization、TMA/WGMMA；
7. 后续推理方向（另项目实现 decode/split-KV）。

在 `docs/.vitepress/config.ts` 中加入文章导航。

### 7.2 更新 CHANGELOG 与 ROADMAP

- `CHANGELOG.md` 增加 `[1.0.0]`：
  - Fixed：benchmark 指标、文档错误、grid.y 限制、int 溢出、非整除测试；
  - 注明验证环境和“未做项”。
- `ROADMAP.md`：
  - 勾选已完成的阶段 1、阶段 4 部分；
  - Phase B/decode 标注为“移出本仓库，在独立推理项目实现”。

### 7.3 打 tag

```bash
git tag -a v1.0.0 -m "v1.0.0: verified reference implementation"
```

### 7.4 T6 验收标准

- [ ] 文章可在 docs 站正常访问，数学公式与代码结论与源码一致。
- [ ] CHANGELOG/ROADMAP 无未勾选但已完成的任务。
- [ ] 仓库有 `v1.0.0` tag。
- [ ] README 定位诚实：不再声称“高性能”，明确“参考实现 + 前向 WMMA”。

---

## 8. C1（可选）：一轮有 profiling 数字的优化

> 只有 T1–T6 全部完成，且仍有时间预算时再做。目标不是变快多少，而是产出一个
> “before/after + 瓶颈证据”的完整叙事。

### 8.1 可选项（三选一，不要全做）

1. **causal 全掩码块特化**：当整个 KV block 都在对角线下方时，跳过逐元素 mask 循环；
2. **WMMA 前向的 K/V cp.async 双缓冲**：隐藏 HBM 延迟；
3. **softmax 行归约 warp 化**：消除“一行一线程”的串行长循环。

### 8.2 执行流程

1. 用 ncu 或 nsys 采基线：
   ```bash
   ncu --set full --launch-count 1 --launch-skip 1 \
     --kernel-name 'regex:flash_attention' \
     ./build/release/cuflash_attn_bench \
     --benchmark_filter='BM_Forward_FP16/4096/64'
   ```
   如果 ncu 无权限，改用 nsys 或至少 CUDA event 对比。
2. 记录优化前数字。
3. 单点修改。
4. 跑全量测试 + benchmark + 记录优化后数字。
5. 写进 `docs/performance/benchmarks.md` 或独立优化笔记。

### 8.3 完成定义

- [ ] 有一组可复现的 before/after 数字；
- [ ] 有一句 profiling 证据说明瓶颈（occupancy、memory latency、bank conflict 等）；
- [ ] 没有为了数字牺牲正确性。

---

## 9. 最终“开发结束”定义（Definition of Done）

本仓库 v1.0 完成即视为该学习项目的主体开发结束：

- [ ] T1–T6 全部完成；
- [ ] `ctest` 全绿，PyTorch 对比通过，benchmark 指标口径正确；
- [ ] README/CHANGELOG/docs/代码一致，无已修复问题残留；
- [ ] 所有性能数字有硬件、基线、复现命令三要素；
- [ ] 技术文章发布；
- [ ] `v1.0.0` tag；
- [ ] Phase B 明确不在本仓库，后续在独立推理项目实施。

达到以上状态后，本仓库冻结为“教学/reference 实现”，不再继续堆功能。
