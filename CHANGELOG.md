# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed
- 面向用户的 GitHub 链接统一为 `github.com/aicl-lab/...`

---

## [0.5.0] - 2026-08-05

### Added

- **Tensor-core forward pass** (`src/forward/flash_attention_forward_wmma.cu`)
  - Phase 2 (forward) of `docs/*/design/tensor-core-migration.md`: FP16
    (sm_70+) and BF16 (sm_80+) forward kernels built on `nvcuda::wmma`
    `m16n16k16` fragments with FP32 accumulation; Q/K/V tiles stay in the
    input precision in shared memory.
  - Runtime dispatch selects the WMMA kernel by compute capability and falls
    back to the scalar kernel on unsupported architectures or missing binaries.
  - BF16 forward/backward benchmarks and PyTorch-comparison cases (guarded by
    device capability).

### Changed

- **BREAKING**: the `L` (logsumexp) parameter of the FP16 and BFloat16
  forward/backward APIs (C++ namespace and C ABI) is now `float*` instead of
  `half*` / `__nv_bfloat16*`. logsumexp is always stored in FP32 because the
  backward pass reconstructs softmax probabilities as `exp(S - L)`; rounding
  `L` to half precision systematically corrupts the gradients. This matches the
  reference FlashAttention, which keeps `softmax_lse` in FP32. Callers that
  passed a reduced-precision `L` buffer must allocate a `float` buffer of shape
  `[batch_size, num_heads, seq_len]` instead.

### Fixed

- Forward/backward launch functions now fall back to smaller tiles at runtime
  when the device's dynamic shared-memory cap (e.g. sm_75, 64 KB) cannot hold
  the primary tiling, instead of returning `CUDA_ERROR`.
- The backward pass allocates its `D` workspace with stream-ordered
  `cudaMallocAsync`/`cudaFreeAsync`, removing a data race when one host thread
  issues backward calls on multiple streams (the old thread-local cache was
  shared across streams).
- The forward online-softmax loop computes `exp()` once per score element
  instead of `HEAD_DIM` times.
- FP16 benchmarks passed a `half*` logsumexp buffer (wrong type and half the
  required size); logsumexp buffers are now allocated as FP32.

### Notes

- Minimum CUDA toolkit is now 11.2 (`cudaMallocAsync`).
- CUDA 13.x removed sm_70 (Volta); build with CUDA 12.x to keep V100 support.

---

## [0.4.0] - 2026-07-28

### ✨ Added | 新增

- **BFloat16 support** (`include/cuflash/flash_attention.h`)
  - BF16 forward and backward passes in both the C++ namespace API and the C ABI
    (`cuflash_attention_{forward,backward}_bf16`).
  - Internal accumulation remains FP32 for numerical stability.
- **Public kernel primitive layer** (`include/cuflash/kernels/`)
  - `matmul`, `online_softmax`, and `tile_io` primitives exposed as a standalone,
    testable API surface (`cuflash::kernels::*`).

### 🔧 Changed | 变更

- Unified the FP32/FP16/BF16 forward and backward kernels into a single dtype
  template (`flash_attention_{forward,backward}_typed.cu`) instead of three
  near-duplicate implementations.
- Adopted FlashAttention-2-style deferred normalization in the forward kernel:
  the running output is kept unnormalized and divided by `l` once at the end,
  removing a per-iteration division.
- Centralized tiling configuration and supported-`head_dim` checks
  (`src/kernels/impl/tile_io.cuh`) as a single source of truth.
- Removed AI control framework directories and stale governance scaffolding
  (`openspec`, `trellis`, `superpowers`, and related tool overlays).
- Simplified contributor guidance, Copilot instructions, and pull request
  metadata to match the real repository workflow.
- Reduced GitHub Pages scope to product documentation; changelog history now
  lives only in the root `CHANGELOG.md`.
- Removed docs-site spec mirrors, release-note mirrors, and AI planning
  artifacts that duplicated repository content.
- Removed stale README references to the deleted `AGENTS.md` workflow document
  and replaced leftover SDD branding with lean repository wording.
- Removed `.claude`/`CLAUDE.local.md` ignore rules that only served deleted AI
  tooling overlays.
- Simplified the GitHub Pages landing page links so the docs site no longer
  surfaces changelog navigation.

### 🐛 Fixed | 修复

- Fixed CUDA preset validation on fresh Ubuntu environments by documenting and
  working through the required host-compiler/toolkit alignment.
- Fixed backward kernel dispatch for `head_dim=64` by using a smaller
  shared-memory tiling path that fits current CUDA limits.
- Fixed package-smoke consumption by exposing CUDA headers through the exported
  target and simplifying the downstream smoke project to a pure C++ consumer.
- Fixed the FP16 tile store test to validate half-rounding semantics instead of
  comparing against the original float with an unrealistically tight tolerance.
- Fixed the optional PyTorch comparison script to skip cleanly when `torch` is
  not installed.

---

## [0.3.0] - 2026-04-24

### 🔧 Chore | 工程整治

#### CI/CD
- Fix `pages.yml`: remove non-existent root `package.json`/`package-lock.json` triggers
- Fix `release.yml`: unify to `cmake --preset release` (remove redundant `-B` flag)
- Fix `docs/.vitepress/config.js`: correct "Specs" nav links from `specs/` to `openspec/specs`

#### Tooling
- Add `.clangd` LSP configuration (CUDA paths, sm_86, diagnostics)
- Add `CMAKE_EXPORT_COMPILE_COMMANDS=ON` for LSP `compile_commands.json` generation
- Update `.vscode/settings.json` with full clangd/CUDA development settings
- Add `.github/copilot-instructions.md` (project-level Copilot instructions, Chinese responses)
- Add `.github/pull_request_template.md`
- Add `.github/ISSUE_TEMPLATE/bug_report.md` (CUDA-specific bug template)

#### Documentation
- Rewrite `AGENTS.md`: high-density CUDA traps, build commands, tool collaboration
- Rewrite `openspec/config.yaml`: Google style (not LLVM), branch strategy, AI tools
- Fix `docs/en/index.md` and `docs/zh/index.md`: update broken `specs/` links to `openspec/specs/`
- Rewrite `CONTRIBUTING.md`: from generic template to project-specific 40-line guide

#### Cleanup
- Delete `QWEN.md` (stale v0.1.0, wrong paths)
- Delete `specs.archived/` (old spec structure, superseded by `openspec/specs/`)
- Delete `CODE_OF_CONDUCT.md` (generic template, no value)

---

## [0.2.0] - 2026-04-16

### 🚀 Highlights | 亮点

This release introduces complete FP16 backward pass support and a thoroughly restructured bilingual documentation system.

本版本引入完整的 FP16 反向传播支持和全面重构的双语文档系统。

### ✨ Added | 新增

#### Features | 功能
- **FP16 Backward Pass** (`src/flash_attention_backward_fp16.cu`)
  - Complete FP16 gradient computation
  - Numerical stability through FP32 internal accumulation
  - C ABI interface for Python ctypes integration
  - Full test coverage

- **Bilingual Documentation System** | 双语文档系统
  - Restructured `docs/en/` and `docs/zh/` directories
  - Professional English API reference with comprehensive examples
  - Detailed algorithm deep dive in both languages
  - New troubleshooting guides for common issues
  - Complete build guides with cross-platform instructions

#### Documentation | 文档
- **API Reference (English)** (`docs/en/api-reference.md`)
  - Complete C++ and C ABI documentation
  - Tensor layout specifications with offset calculations
  - Error handling examples and best practices
  - Thread safety and memory management guidelines
  - GPU architecture support matrix

- **Algorithm Documentation** (`docs/en/algorithm.md`, `docs/zh/algorithm.md`)
  - Standard attention bottleneck explanation
  - Core FlashAttention concepts (tiling, online softmax, recomputation)
  - Step-by-step forward and backward algorithms
  - Causal masking strategy with efficiency analysis
  - FP16 implementation details and precision handling
  - Memory complexity analysis with real-world comparisons

- **Build Guide** (`docs/en/building.md`, `docs/zh/building.md`)
  - CMake presets for common configurations
  - Cross-platform build instructions (Linux, Windows, Docker)
  - GPU architecture targeting guide
  - Troubleshooting common build issues

- **Troubleshooting Guide** (`docs/en/troubleshooting.md`, `docs/zh/troubleshooting.md`)
  - Build issue resolution
  - Runtime error diagnosis
  - Performance optimization tips
  - Numerical accuracy considerations
  - Complete error code reference

### 🔧 Changed | 变更

- **Project Documentation Restructure**
  - Migrated all docs to language-specific subdirectories
  - Updated HonKit configuration for bilingual support
  - Enhanced navigation and cross-references

- **README Updates**
  - Professionalized English README with badges and clear structure
  - Synchronized Chinese README with consistent formatting
  - Added quick start examples and feature highlights
  - Improved project structure visualization

### 🐛 Fixed | 修复

- None in this release (all fixes included in previous versions)

### 🗑️ Removed | 移除

- Deleted legacy `docs/*.md` files (migrated to new structure)
- Removed unused `src/cuflash_ctypes_api.cu` (duplicate C ABI)

---

## [0.1.0] - 2026-03-13

### ✨ Added | 新增

- **Complete Documentation Suite**
  - API reference documentation (`docs/api.md`)
  - FlashAttention algorithm deep dive (`docs/algorithm.md`)
  - Build guide with CMake instructions (`docs/building.md`)
  - HonKit documentation site with Chinese search support

### 🔧 Changed | 变更

- **CI Workflow Improvement**
  - Switched to CPU-safe mode for CI (format checking only)
  - Reason: GitHub Hosted Runners don't provide GPU, CUDA builds were failing
  - Now runs clang-format static checks only

### 🔨 CI/CD | 持续集成

- Unified GitHub Actions workflow naming conventions
- Added `permissions` and `concurrency` configurations
- Pages workflow with `paths` filtering and sparse-checkout

---

## [0.1.0-alpha.2] - 2026-03-10

### ✨ Added | 新增

- **Standardized CI Workflow** (`.github/workflows/ci.yml`)
  - Support for `push`, `pull_request`, `workflow_dispatch` triggers
  - Added clang-format code format checking

### 🔧 Changed | 变更

- Renamed Pages workflow: `docs.yml` → `pages.yml`

---

## [0.1.0-alpha.1] - 2026-02-13

### ✨ Added | 新增

- **FP16 Forward Pass** (`src/flash_attention_fp16.cu`)
  - FP16 forward kernel implementation
  - FP16 API integration with full `half` type support

### ⚡ Performance Optimizations | 性能优化

- **Vectorized Memory Access**
  - `float4` vectorized loads/stores for improved global memory bandwidth
  
- **Launch Bounds**
  - Added `__launch_bounds__(128)` to all CUDA kernels for occupancy optimization
  
- **Fast Math Option**
  - Optional `--use_fast_math` compiler flag for faster execution

### 🐛 Bug Fixes | 错误修复

**Stream Safety in Backward Pass:**
- Changed `cudaMemset` to `cudaMemsetAsync` for in-stream ordering
- Added `cudaStreamSynchronize` to prevent premature workspace deallocation
- Added `cudaMalloc` return value checking

### 🏗️ Build Configuration | 构建配置

- Added SM 89 (Ada Lovelace, RTX 4090) support
- Added SM 90 (Hopper, H100) support

---

## Version Summary | 版本摘要

| Version | Key Features | Release Date |
|---------|--------------|--------------|
| [0.5.0] | Tensor-core (WMMA) forward, FP32 logsumexp ABI, sm_75 tiling fallback, stream-ordered workspace | 2026-08-05 |
| [0.4.0] | BF16 support, unified dtype templates, FA2 deferred normalization, public kernel primitives | 2026-07-28 |
| [0.3.0] | Engineering cleanup, clangd/LSP tooling, contribution guides | 2026-04-24 |
| [0.2.0] | FP16 backward, bilingual docs, troubleshooting guide | 2026-04-16 |
| [0.1.0] | Complete docs, CPU-safe CI, HonKit site | 2026-03-13 |
| [0.1.0-alpha.2] | Standardized CI, format checking | 2026-03-10 |
| [0.1.0-alpha.1] | FP16 forward, performance optimizations | 2026-02-13 |

---

## Release Links | 发布链接

- [v0.2.0](https://github.com/aicl-lab/cuflash-attn/releases/tag/v0.2.0)
- [v0.1.0](https://github.com/aicl-lab/cuflash-attn/releases/tag/v0.1.0)
- [v0.1.0-alpha.2](https://github.com/aicl-lab/cuflash-attn/releases/tag/v0.1.0-alpha.2)
- [v0.1.0-alpha.1](https://github.com/aicl-lab/cuflash-attn/releases/tag/v0.1.0-alpha.1)

---

[Unreleased]: https://github.com/aicl-lab/cuflash-attn/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/aicl-lab/cuflash-attn/compare/v0.4.0...v0.5.0
[0.2.0]: https://github.com/aicl-lab/cuflash-attn/releases/tag/v0.2.0
[0.1.0]: https://github.com/aicl-lab/cuflash-attn/releases/tag/v0.1.0
[0.1.0-alpha.2]: https://github.com/aicl-lab/cuflash-attn/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/aicl-lab/cuflash-attn/releases/tag/v0.1.0-alpha.1
