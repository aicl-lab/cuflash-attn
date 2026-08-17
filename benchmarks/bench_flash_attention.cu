#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuflash/flash_attention.h"
#include "impl/tile_io.cuh"
#include "kernel_launch_utils.cuh"

// Helper: allocate device memory and fill with random data
template<typename T>
static std::vector<T*> allocate_and_init(const std::vector<size_t>& sizes) {
    std::vector<T*> ptrs;
    ptrs.reserve(sizes.size());
    for (size_t size : sizes) {
        T* d_ptr = nullptr;
        size_t bytes = size * sizeof(T);
        cudaMalloc(&d_ptr, bytes);
        // Fill with random values in [-1, 1]
        std::vector<T> h_data(size);
        for (size_t i = 0; i < size; ++i) {
            h_data[i] = static_cast<T>(
                2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 1.0f);
        }
        cudaMemcpy(d_ptr, h_data.data(), bytes, cudaMemcpyHostToDevice);
        ptrs.push_back(d_ptr);
    }
    return ptrs;
}

// =============================================================================
// FLOPs and logical-HBM models
// =============================================================================
// FLOPs model (scalar and tensor-core forward, scalar backward):
//   forward: 2*N^2*D (QK^T) + 2*N^2*D (PV) = 4*B*H*N^2*D
//   causal forward ~= half.
//   backward (this implementation's actual scalar path):
//     dQ kernel  3 tile matmuls (QK^T, dO·V^T, dS·K)
//     dKdV kernel 4 tile matmuls (QK^T, P^T·dO, dO·V^T, dS^T·Q)
//     = 14 * B*H*N^2*D FLOPs; causal ~= half.
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

// Logical HBM elements for the forward pass (no padding counted):
//   Q read once + O written once; K and V are re-read once per Q block.
//   causal: a Q block only touches the KV rows up to its last row.
static double fwd_logical_hbm_elements(int bh, int n, int d, bool causal, int bm) {
    const int q_blocks = (n + bm - 1) / bm;
    double kv_rows = 0.0;
    for (int qb = 0; qb < q_blocks; ++qb) {
        const int q_start = qb * bm;
        const int valid_rows = causal ? std::min(q_start + bm, n) : n;
        kv_rows += static_cast<double>(valid_rows);
    }
    // Q + O 各 1 次，K/V 各按每个 Q block 的重载次数计
    return 2.0 * static_cast<double>(bh) * n * d +
           2.0 * static_cast<double>(bh) * d * kv_rows;
}

// Forward tiling choices, mirroring the launcher in
// flash_attention_forward_typed.cu / flash_attention_forward_wmma.cu.
struct FwdTile {
    int bm = 0;
    int bn = 0;
};

static FwdTile fwd_scalar_tile(int head_dim, int max_dynamic_smem) {
    using C = cuflash::impl::ForwardTilingConfig;
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

static FwdTile fwd_wmma_tile(int head_dim) {
    return (head_dim == 128) ? FwdTile{32, 32} : FwdTile{64, 32};
}

// Pick the tile the forward launcher will actually use for a given element
// size: FP32 always takes the scalar path; reduced precision takes the WMMA
// path on sm_70+ (FP16) / sm_80+ (BF16) and the scalar path otherwise.
static FwdTile fwd_tile_for_elem_size(size_t elem_size, int head_dim, int max_dynamic_smem) {
    if (elem_size == sizeof(float)) {
        return fwd_scalar_tile(head_dim, max_dynamic_smem);
    }
    int device = 0;
    int major = 0;
    bool use_wmma = false;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) ==
            cudaSuccess) {
        // On sm_8x+ both FP16 and BF16 take the WMMA path; on sm_70 FP16 does.
        use_wmma = major >= 8 || major == 7;
    }
    return use_wmma ? fwd_wmma_tile(head_dim) : fwd_scalar_tile(head_dim, max_dynamic_smem);
}

// Report achieved compute and (forward-only) logical HBM throughput so results
// can be read against the hardware roofline.
//   Logical HBM is reported for the forward pass only; backward HBM would need
//   an ncu measurement because the current code has no reliable byte model.
//   Counter names use "LogicalHBM GB/s" on purpose so nobody mistakes the
//   modeled number for a measured HBM bandwidth.
static void report_metrics(benchmark::State& state, int batch_size, int num_heads,
                           int seq_len, int head_dim, size_t elem_size,
                           bool backward, bool causal) {
    const double flops =
        attention_flops(batch_size, num_heads, seq_len, head_dim, backward, causal);
    state.counters["GFLOPS/s"] = benchmark::Counter(
        flops / 1e9,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);

    if (!backward) {
        int max_dynamic_smem = 0;
        cuflash::query_max_dynamic_shared_memory_per_block(&max_dynamic_smem);
        const FwdTile tile = fwd_tile_for_elem_size(elem_size, head_dim, max_dynamic_smem);
        const double logical_bytes =
            fwd_logical_hbm_elements(batch_size * num_heads, seq_len, head_dim, causal, tile.bm) *
            static_cast<double>(elem_size);
        state.counters["LogicalHBM GB/s"] = benchmark::Counter(
            logical_bytes / 1e9,
            benchmark::Counter::kIsIterationInvariantRate,
            benchmark::Counter::OneK::kIs1000);
    }
}

// =============================================================================
// Naive (materialized) attention baseline
// =============================================================================
// Forms the full N x N score matrix. This is the "standard" attention that
// FlashAttention exists to avoid; it is included ONLY as a comparison point
// (and intentionally runs out of memory at large N), not as a recommended
// implementation. Timed with wall-clock + cudaStreamSynchronize: this is NOT
// a production baseline and is intentionally not event-timed.
__global__ void naive_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                       const float* __restrict__ V, float* __restrict__ O,
                                       int seq_len, int head_dim, float scale, bool causal) {
    const int bh = blockIdx.y;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    if (row >= seq_len)
        return;

    const float* Q_row =
        Q + static_cast<size_t>(bh) * seq_len * head_dim + static_cast<size_t>(row) * head_dim;
    const float* K_base = K + static_cast<size_t>(bh) * seq_len * head_dim;
    const float* V_base = V + static_cast<size_t>(bh) * seq_len * head_dim;
    float* O_row =
        O + static_cast<size_t>(bh) * seq_len * head_dim + static_cast<size_t>(row) * head_dim;

    extern __shared__ float scores[];  // seq_len floats
    __shared__ float red[128];

    // scores[j] = scale * <Q_row, K_j>, with the causal mask applied.
    float local_max = -INFINITY;
    for (int j = tid; j < seq_len; j += nthreads) {
        float s = -INFINITY;
        if (!(causal && j > row)) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q_row[d] * K_base[static_cast<size_t>(j) * head_dim + d];
            }
            s = dot * scale;
        }
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }

    red[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s)
            red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float row_max = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += nthreads) {
        float e = expf(scores[j] - row_max);
        scores[j] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s)
            red[tid] += red[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / red[0];
    __syncthreads();

    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += scores[j] * V_base[static_cast<size_t>(j) * head_dim + d];
        }
        O_row[d] = acc * inv_sum;
    }
}

static void BM_NaiveForward_FP32(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;

    auto devs = allocate_and_init<float>({qkv_size, qkv_size, qkv_size, qkv_size});  // Q, K, V, O
    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2], *d_O = devs[3];

    dim3 grid(seq_len, batch_size * num_heads);
    size_t smem = static_cast<size_t>(seq_len) * sizeof(float);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    for (auto _ : state) {
        naive_attention_kernel<<<grid, 128, smem, stream>>>(d_Q, d_K, d_V, d_O, seq_len, head_dim,
                                                            scale, false);
        cudaStreamSynchronize(stream);
    }

    // The naive path also materializes the N x N score matrix (written + read),
    // so its HBM model is intentionally different from the FlashAttention path:
    //   Q/K/V/O each once + S/P written and read once = 4*elems + 2*B*H*N^2 elems.
    const double flops =
        attention_flops(batch_size, num_heads, seq_len, head_dim, /*backward=*/false,
                        /*causal=*/false);
    const double elems = static_cast<double>(batch_size) * num_heads * seq_len * head_dim;
    const double bytes = 4.0 * elems * sizeof(float) +
                         2.0 * static_cast<double>(batch_size) * num_heads * seq_len * seq_len *
                             sizeof(float);
    state.counters["GFLOPS/s"] = benchmark::Counter(
        flops / 1e9,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);
    state.counters["HBM GB/s"] = benchmark::Counter(
        bytes / 1e9,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000);

    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
}
// Keep N modest: the baseline allocates an O(N^2) score matrix.
BENCHMARK(BM_NaiveForward_FP32)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Unit(benchmark::kMillisecond);

// =============================================================================
// FlashAttention forward / backward (CUDA Event timed)
// =============================================================================
// All FlashAttention benchmarks below use CUDA events on the launch stream so
// that the reported iteration time excludes host-side overhead. Do NOT
// cudaMalloc/cudaFree or re-initialize data inside the timing loop.

// FP32 Forward Benchmark
static void BM_Forward_FP32(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    auto devs = allocate_and_init<float>({qkv_size, qkv_size, qkv_size,  // Q, K, V
                                          qkv_size,                      // O
                                          l_size});                      // L

    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (auto _ : state) {
        cudaEventRecord(ev_start, stream);
        auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                    seq_len, head_dim, scale, false, stream);
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

    report_metrics(state, batch_size, num_heads, seq_len, head_dim, sizeof(float),
                   /*backward=*/false, /*causal=*/false);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
}
BENCHMARK(BM_Forward_FP32)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// FP32 Backward Benchmark
static void BM_Backward_FP32(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    // Q, K, V, O, L, dO, dQ, dK, dV
    auto devs = allocate_and_init<float>(
        {qkv_size, qkv_size, qkv_size, qkv_size, l_size, qkv_size, qkv_size, qkv_size, qkv_size});

    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2], *d_O = devs[3];
    float *d_L = devs[4], *d_dO = devs[5], *d_dQ = devs[6], *d_dK = devs[7];
    float* d_dV = devs[8];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (auto _ : state) {
        cudaEventRecord(ev_start, stream);
        auto err = cuflash::flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK,
                                                     d_dV, batch_size, num_heads, seq_len, head_dim,
                                                     scale, false, stream);
        cudaEventRecord(ev_stop, stream);

        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_backward failed");
            break;
        }

        cudaEventSynchronize(ev_stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        state.SetIterationTime(elapsed_ms / 1000.0);
    }

    report_metrics(state, batch_size, num_heads, seq_len, head_dim, sizeof(float),
                   /*backward=*/true, /*causal=*/false);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
}
BENCHMARK(BM_Backward_FP32)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Reduced-precision (FP16/BF16) forward. L (logsumexp) is always FP32 even for
// reduced-precision inputs, so it is allocated separately; see
// include/cuflash/flash_attention.h for the rationale.
template<typename InputT>
static void BM_Forward_ReducedPrec(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    auto devs = allocate_and_init<InputT>({qkv_size, qkv_size, qkv_size,  // Q, K, V
                                           qkv_size});                    // O
    auto l_bufs = allocate_and_init<float>({l_size});                     // L (FP32)

    InputT *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    InputT* d_O = devs[3];
    float* d_L = l_bufs[0];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (auto _ : state) {
        cudaEventRecord(ev_start, stream);
        auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                    seq_len, head_dim, scale, false, stream);
        cudaEventRecord(ev_stop, stream);

        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_forward (reduced precision) failed");
            break;
        }

        cudaEventSynchronize(ev_stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        state.SetIterationTime(elapsed_ms / 1000.0);
    }

    report_metrics(state, batch_size, num_heads, seq_len, head_dim, sizeof(InputT),
                   /*backward=*/false, /*causal=*/false);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
    for (auto* ptr : l_bufs) {
        cudaFree(ptr);
    }
}

static void BM_Forward_FP16(benchmark::State& state) {
    BM_Forward_ReducedPrec<half>(state);
}
BENCHMARK(BM_Forward_FP16)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

static void BM_Forward_BF16(benchmark::State& state) {
    BM_Forward_ReducedPrec<__nv_bfloat16>(state);
}
BENCHMARK(BM_Forward_BF16)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Reduced-precision (FP16/BF16) backward with FP32 L, same rationale as above.
template<typename InputT>
static void BM_Backward_ReducedPrec(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    // Q, K, V, O, dO, dQ, dK, dV
    auto devs = allocate_and_init<InputT>(
        {qkv_size, qkv_size, qkv_size, qkv_size, qkv_size, qkv_size, qkv_size, qkv_size});
    auto l_bufs = allocate_and_init<float>({l_size});  // L (FP32)

    InputT *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2], *d_O = devs[3];
    InputT *d_dO = devs[4], *d_dQ = devs[5], *d_dK = devs[6];
    InputT* d_dV = devs[7];
    float* d_L = l_bufs[0];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (auto _ : state) {
        cudaEventRecord(ev_start, stream);
        auto err = cuflash::flash_attention_backward(d_Q, d_K, d_V, d_O, d_L, d_dO, d_dQ, d_dK,
                                                     d_dV, batch_size, num_heads, seq_len, head_dim,
                                                     scale, false, stream);
        cudaEventRecord(ev_stop, stream);

        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_backward (reduced precision) failed");
            break;
        }

        cudaEventSynchronize(ev_stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        state.SetIterationTime(elapsed_ms / 1000.0);
    }

    report_metrics(state, batch_size, num_heads, seq_len, head_dim, sizeof(InputT),
                   /*backward=*/true, /*causal=*/false);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
    for (auto* ptr : l_bufs) {
        cudaFree(ptr);
    }
}

static void BM_Backward_FP16(benchmark::State& state) {
    BM_Backward_ReducedPrec<half>(state);
}
BENCHMARK(BM_Backward_FP16)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

static void BM_Backward_BF16(benchmark::State& state) {
    BM_Backward_ReducedPrec<__nv_bfloat16>(state);
}
BENCHMARK(BM_Backward_BF16)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Causal Mask Forward Benchmark
static void BM_Forward_Causal(benchmark::State& state) {
    int seq_len = state.range(0);
    int head_dim = state.range(1);
    int batch_size = 1;
    int num_heads = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    auto devs = allocate_and_init<float>({qkv_size, qkv_size, qkv_size,  // Q, K, V
                                          qkv_size,                      // O
                                          l_size});                      // L

    float *d_Q = devs[0], *d_K = devs[1], *d_V = devs[2];
    float *d_O = devs[3], *d_L = devs[4];

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    for (auto _ : state) {
        cudaEventRecord(ev_start, stream);
        auto err = cuflash::flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads,
                                                    seq_len, head_dim, scale, true, stream);
        cudaEventRecord(ev_stop, stream);

        if (err != cuflash::FlashAttentionError::SUCCESS) {
            state.SkipWithError("flash_attention_forward (causal) failed");
            break;
        }

        cudaEventSynchronize(ev_stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
        state.SetIterationTime(elapsed_ms / 1000.0);
    }

    report_metrics(state, batch_size, num_heads, seq_len, head_dim, sizeof(float),
                   /*backward=*/false, /*causal=*/true);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    for (auto* ptr : devs) {
        cudaFree(ptr);
    }
}
BENCHMARK(BM_Forward_Causal)
    ->Args({256, 64})
    ->Args({512, 64})
    ->Args({1024, 64})
    ->Args({2048, 64})
    ->Args({4096, 64})
    ->Args({4096, 128})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_MAIN();
