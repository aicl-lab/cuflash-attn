# 架构总览

本页提供 CuFlash-Attn 的全面架构视图，面向需要理解系统设计的研究人员和工程师。

---

## 系统架构

```mermaid
graph TB
    subgraph API["API 层"]
        CPP["C++ 命名空间 API<br/>cuflash::flash_attention_*"]
        CABI["C ABI<br/>cuflash_attention_*"]
        PY["Python 绑定<br/>ctypes 接口"]
    end

    subgraph DISPATCH["调度层"]
        VALID["参数验证"]
        SELECT["内核选择<br/>按数据类型和架构"]
        LAUNCH["内核启动"]
    end

    subgraph KERNEL["内核层"]
        FWD["前向内核<br/>flash_attention_forward_*"]
        BWD["反向内核<br/>flash_attention_backward_*"]
        UTIL["工具函数<br/>softmax, 归约"]
    end

    subgraph HW["硬件层"]
        SM["共享内存<br/>SRAM 分块"]
        REG["寄存器堆<br/>累加器"]
        HBM["HBM<br/>全局内存"]
    end

    CPP --> VALID
    CABI --> VALID
    PY --> VALID

    VALID --> SELECT
    SELECT --> LAUNCH
    LAUNCH --> FWD
    LAUNCH --> BWD

    FWD --> SM
    FWD --> REG
    BWD --> SM
    BWD --> REG
    UTIL --> SM

    SM --> HBM
    REG --> HBM
```

---

## 数据流

### 前向传播

```mermaid
sequenceDiagram
    participant Host as 主机
    participant GPU as GPU 内核
    participant SRAM as 共享内存
    participant REG as 寄存器

    Host->>GPU: 启动内核 (Q, K, V)
    
    loop 对每个分块
        GPU->>SRAM: 加载 Q_tile, K_tile, V_tile
        SRAM->>REG: 计算 QK^T (部分)
        REG->>REG: 在线 softmax 更新
        REG->>REG: 累加 O_partial
    end
    
    REG->>GPU: 写入 O, L (log-sum-exp)
    GPU->>Host: 返回
```

### 反向传播

```mermaid
sequenceDiagram
    participant Host as 主机
    participant GPU as GPU 内核
    participant SRAM as 共享内存
    participant REG as 寄存器

    Host->>GPU: 启动内核 (Q, K, V, O, L, dO)
    
    Note over GPU: 按需重计算注意力
    
    loop 对每个分块
        GPU->>SRAM: 加载分块
        SRAM->>REG: 计算 dQ, dK, dV 部分
        REG->>REG: 累加梯度
    end
    
    REG->>GPU: 写入 dQ, dK, dV
    GPU->>Host: 返回
```

---

## 内存布局

```mermaid
graph LR
    subgraph Input["输入张量"]
        Q["Q: [B, H, N, D]"]
        K["K: [B, H, N, D]"]
        V["V: [B, H, N, D]"]
    end

    subgraph Output["输出张量"]
        O["O: [B, H, N, D]"]
        L["L: [B, H, N]"]
    end

    subgraph Gradients["梯度张量"]
        dO["dO: [B, H, N, D]"]
        dQ["dQ: [B, H, N, D]"]
        dK["dK: [B, H, N, D]"]
        dV["dV: [B, H, N, D]"]
    end

    Q --> O
    K --> O
    V --> O
    Q --> L
    K --> L

    O --> dQ
    O --> dK
    O --> dV
    dO --> dQ
    dO --> dK
    dO --> dV
```

---

## 内核分块策略

### 分块维度

前向 tile 集中定义在 `src/kernels/impl/tile_io.cuh`（`ForwardTilingConfig`）：

| 参数 | 描述 | scalar 前向 | WMMA 前向 |
|------|------|-------------|-----------|
| `B_r` | Query 分块大小 | 64（hd128 为 32） | 64（hd128 为 32） |
| `B_c` | Key/Value 分块大小 | 64（hd128 为 32） | 32 |
| `D` | 头维度 | 32, 64, 128 | 32, 64, 128 |
| `T_r` | 每 Query 分块线程数 | 128 | 128 |

反向使用更保守的 `BackwardTilingConfig`（shared memory 要容纳更多梯度张量），例如
hd128 为 $16 \times 32$，hd64 为 $32 \times 32$。

### 内存复杂度

$$
\text{SRAM} = O(B_r \times D + B_c \times D + B_r \times B_c)
$$

例如 scalar 前向 ($B_r=64, B_c=64, D=64$，FP32)：

$$
\text{SRAM 元素} = 64 \times 64 \;(Q) + 64 \times 64 \;(K) + 64 \times 64 \;(V) + 64 \times 64 \;(S) + 64 \times 64 \;(O) + 64 + 64 = 20608
$$

即 $\approx 80$ KB（超过默认 48 KB 上限时由 launcher opt-in 动态共享内存），精确值见
`ForwardTilingConfig::smem_bytes`。

---

## 目录结构

```
cuflash-attn/
├── include/cuflash/          # 公开 API 头文件
│   ├── flash_attention.h     # C++ 命名空间 API（含 C ABI 声明）
│   ├── export.h              # 可见性宏
│   └── version.h.in          # 版本头文件模板
├── src/
│   ├── api/                  # API 调度层
│   │   └── flash_attention_api.cu
│   ├── forward/              # 前向内核（统一模板）
│   │   ├── flash_attention_forward_typed.cu   # FP32/FP16/BF16 scalar 前向
│   │   └── flash_attention_forward_wmma.cu    # FP16/BF16 WMMA（Tensor Core）前向
│   ├── backward/             # 反向内核（统一模板，scalar）
│   │   └── flash_attention_backward_typed.cu
│   └── kernels/              # 共享工具
│       ├── impl/             # 内部实现细节（tiling 配置、online softmax、type adapter）
│       ├── matmul.cu / online_softmax.cu / tile_io.cu
│       └── kernel_launch_utils.cuh
└── tests/
    ├── unit/                  # 单元测试（gtest）
    ├── integration/           # API smoke + PyTorch 对比
    └── package_smoke/         # 安装包冒烟
```

---

## 错误处理流程

```mermaid
graph TD
    INPUT[API 调用] --> VALID{验证参数}
    VALID -->|无效| ERR[返回错误码]
    VALID -->|有效| ALLOC{分配内存}
    ALLOC -->|失败| ERR
    ALLOC -->|成功| LAUNCH[启动内核]
    LAUNCH -->|CUDA 错误| ERR
    LAUNCH -->|成功| SYNC[同步流]
    SYNC -->|CUDA 错误| ERR
    SYNC -->|成功| SUCCESS[返回 SUCCESS]
```

---

## 性能特征

| 操作 | 内存 | 计算 | 带宽受限 |
|------|------|------|----------|
| 前向 | $O(N)$ | $O(N^2)$ | 是 (低 D) |
| 反向 | $O(N)$ | $O(N^2)$ | 是 (低 D) |
| 重计算 | $O(1)$ | $O(N^2)$ | 是 |

::: tip 关键洞察
FlashAttention 通过永不物化完整注意力矩阵，将内存从 $O(N^2)$ 降至 $O(N)$。代价是在反向传播时重计算注意力分数，这是计算受限的操作，因此在现代 GPU 上效率很高。
:::
