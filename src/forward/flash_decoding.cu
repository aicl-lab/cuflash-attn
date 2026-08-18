// FlashDecoding (Split-KV) forward kernel for decode (query_len == 1).
//
// Story: in decode, each step attends one query against the whole KV cache.
// For long sequences that KV sweep is sequential in the classic kernel, so
// we split the KV dimension into `num_chunks` independent blocks:
//   Phase 1 (flash_decoding_partial_kernel): block (chunk, batch*head)
//     computes the online-softmax partial (m, l, unnormalized O) over its
//     chunk of KV rows.
//   Phase 2 (flash_decoding_combine_kernel): reduces the partials across
//     chunks with the flash-attention rescaling rule:
//       m = max_c m_c,  l = sum_c l_c * exp(m_c - m)
//       O = sum_c O_c * exp(m_c - m) / l
//
// Layout (same as forward):
//   Q: [batch_size * num_heads, 1, head_dim]
//   K/V: [batch_size * num_heads, seq_len, head_dim]
//   O: [batch_size * num_heads, 1, head_dim]
//   L: [batch_size * num_heads]  (logsumexp = m + log(l))
//
// Decode semantics: single query attends all KV rows (causal decode over
// [0, seq_len) with query at position seq_len-1 is the same set of keys).
// FP32 accumulation; output written through impl::TypeAdapter so the same
// kernel serves FP32 / FP16 / BF16.

#include <float.h>
#include <math.h>

#include <type_traits>

#include "cuflash/flash_attention.h"
#include "impl/type_adapter.cuh"
#include "kernel_launch_utils.cuh"

namespace cuflash {

namespace {

constexpr int kDecodeThreads = 128;
constexpr int kDecodeBlockN = 64;

// Block reduction helpers over kDecodeThreads threads (one warp reduce +
// cross-warp shared scratch of 4 floats, 128 threads = 4 warps).
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset);
    return val;
}

template <bool IsMax>
__device__ __forceinline__ float block_reduce(float val, float* red) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    constexpr int kWarps = kDecodeThreads / 32;
    val = IsMax ? warp_reduce_max(val) : warp_reduce_sum(val);
    if (lane == 0) red[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < kWarps) ? red[lane] : (IsMax ? -FLT_MAX : 0.0f);
        val = IsMax ? warp_reduce_max(val) : warp_reduce_sum(val);
        if (lane == 0) red[0] = val;
    }
    __syncthreads();
    return red[0];
}

// Phase 1: partial online softmax over one KV chunk for one (batch, head).
template <typename InputT, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(kDecodeThreads)
    flash_decoding_partial_kernel(const InputT* __restrict__ Q, const InputT* __restrict__ K,
                                  const InputT* __restrict__ V, float* __restrict__ partial_O,
                                  float* __restrict__ partial_m, float* __restrict__ partial_l,
                                  int seq_len, int num_chunks, int chunk_len, float scale) {
    const int bh = blockIdx.y;
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;

    const int chunk_start = chunk * chunk_len;
    const int chunk_size = min(chunk_len, seq_len - chunk_start);
    const int partial_idx = chunk * gridDim.y + bh;

    const InputT* Q_ptr = Q + static_cast<size_t>(bh) * HEAD_DIM;
    const InputT* K_ptr = K + static_cast<size_t>(bh) * seq_len * HEAD_DIM;
    const InputT* V_ptr = V + static_cast<size_t>(bh) * seq_len * HEAD_DIM;

    extern __shared__ float smem[];
    float* q_s = smem;                            // HEAD_DIM
    float* K_tile = q_s + HEAD_DIM;               // BLOCK_N * HEAD_DIM
    float* V_tile = K_tile + BLOCK_N * HEAD_DIM;  // BLOCK_N * HEAD_DIM
    float* scores = V_tile + BLOCK_N * HEAD_DIM;  // BLOCK_N
    float* red = scores + BLOCK_N;                // 4 floats (kDecodeThreads/32)

    for (int d = tid; d < HEAD_DIM; d += kDecodeThreads)
        q_s[d] = impl::TypeAdapter<InputT>::to_compute(Q_ptr[d]);
    __syncthreads();

    float m = -FLT_MAX;
    float l = 0.0f;
    float o_acc[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) o_acc[d] = 0.0f;

    for (int kv = chunk_start; kv < chunk_start + chunk_size; kv += BLOCK_N) {
        const int tile_n = min(BLOCK_N, chunk_start + chunk_size - kv);

        for (int i = tid; i < tile_n * HEAD_DIM; i += kDecodeThreads) {
            const int j = i / HEAD_DIM;
            const int d = i % HEAD_DIM;
            K_tile[i] = impl::TypeAdapter<InputT>::to_compute(
                K_ptr[static_cast<size_t>(kv + j) * HEAD_DIM + d]);
            V_tile[i] = impl::TypeAdapter<InputT>::to_compute(
                V_ptr[static_cast<size_t>(kv + j) * HEAD_DIM + d]);
        }
        __syncthreads();

        // Scores: thread j computes dot(q, K[j]) for j < tile_n.
        for (int j = tid; j < tile_n; j += kDecodeThreads) {
            float s = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) s += q_s[d] * K_tile[j * HEAD_DIM + d];
            scores[j] = s * scale;
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        for (int j = tid; j < tile_n; j += kDecodeThreads) tile_max = fmaxf(tile_max, scores[j]);
        tile_max = block_reduce<true>(tile_max, red);

        const float m_new = fmaxf(m, tile_max);
        const float rescale = expf(m - m_new);

        float sum_exp = 0.0f;
        for (int j = tid; j < tile_n; j += kDecodeThreads)
            sum_exp += expf(scores[j] - m_new);
        sum_exp = block_reduce<false>(sum_exp, red);

        for (int d = tid; d < HEAD_DIM; d += kDecodeThreads) {
            float partial = 0.0f;
            for (int j = 0; j < tile_n; ++j)
                partial += expf(scores[j] - m_new) * V_tile[j * HEAD_DIM + d];
            o_acc[d] = o_acc[d] * rescale + partial;
        }

        l = l * rescale + sum_exp;
        m = m_new;
        __syncthreads();
    }

    partial_m[partial_idx] = m;
    partial_l[partial_idx] = l;
    for (int d = tid; d < HEAD_DIM; d += kDecodeThreads)
        partial_O[static_cast<size_t>(partial_idx) * HEAD_DIM + d] = o_acc[d];
}

// Phase 2: reduce partials across chunks (one block per batch*head).
template <typename InputT, int HEAD_DIM>
__global__ void __launch_bounds__(kDecodeThreads)
    flash_decoding_combine_kernel(const float* __restrict__ partial_O,
                                  const float* __restrict__ partial_m,
                                  const float* __restrict__ partial_l,
                                  InputT* __restrict__ O, float* __restrict__ L, int num_chunks,
                                  int bh_total) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;

    // Partial 布局与 phase 1 一致：[num_chunks][bh_total]（stride = bh_total）。
    float m = -FLT_MAX;
    for (int c = tid; c < num_chunks; c += kDecodeThreads)
        m = fmaxf(m, partial_m[c * bh_total + bh]);
    extern __shared__ float red_s[];
    m = block_reduce<true>(m, red_s);

    float l = 0.0f;
    for (int c = tid; c < num_chunks; c += kDecodeThreads)
        l += partial_l[c * bh_total + bh] * expf(partial_m[c * bh_total + bh] - m);
    l = block_reduce<false>(l, red_s);

    for (int d = tid; d < HEAD_DIM; d += kDecodeThreads) {
        float o = 0.0f;
        for (int c = 0; c < num_chunks; ++c)
            o += partial_O[(static_cast<size_t>(c) * bh_total + bh) * HEAD_DIM + d] *
                 expf(partial_m[c * bh_total + bh] - m);
        O[static_cast<size_t>(bh) * HEAD_DIM + d] =
            impl::TypeAdapter<InputT>::from_compute(o / l);
    }
    if (tid == 0) L[bh] = m + logf(l);
}

// Dispatch helper per dtype / head_dim.
template <typename InputT>
FlashAttentionError launch_flash_decoding_typed(const InputT* Q, const InputT* K, const InputT* V,
                                                InputT* O, float* L, int batch_size, int num_heads,
                                                int seq_len, int head_dim, float scale,
                                                int num_chunks, cudaStream_t stream) {
    if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr || L == nullptr) {
        return FlashAttentionError::NULL_POINTER;
    }
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        return FlashAttentionError::INVALID_DIMENSION;
    }
    if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return FlashAttentionError::UNSUPPORTED_HEAD_DIM;
    }
    if (num_chunks <= 0) num_chunks = 1;
    const int max_chunks = (seq_len + kDecodeBlockN - 1) / kDecodeBlockN;
    num_chunks = min(num_chunks, max_chunks);
    const int chunk_len = (seq_len + num_chunks - 1) / num_chunks;

    const int bh_total = batch_size * num_heads;

    // Scratch for partials: [num_chunks, bh_total] for m/l, [num_chunks, bh_total, D] for O.
    // 函数级 static：避免 decode 每步 cudaMalloc；不够大时重新分配（单流假设）。
    const size_t partial_m_size = static_cast<size_t>(num_chunks) * bh_total;
    const size_t partial_o_size = partial_m_size * head_dim;
    const size_t need_bytes = (partial_o_size + 2 * partial_m_size) * sizeof(float);
    static float* scratch = nullptr;
    static size_t scratch_bytes = 0;
    if (scratch == nullptr || scratch_bytes < need_bytes) {
        if (scratch != nullptr) cudaFree(scratch);
        if (cudaMalloc(&scratch, need_bytes) != cudaSuccess) {
            scratch = nullptr;
            return FlashAttentionError::CUDA_ERROR;
        }
        scratch_bytes = need_bytes;
    }
    float* partial_o = scratch;
    float* partial_m = partial_o + partial_o_size;
    float* partial_l = partial_m + partial_m_size;

    dim3 grid(num_chunks, bh_total);
    size_t shared_partial = (kDecodeBlockN * 2 * head_dim + head_dim + kDecodeBlockN + 4) * sizeof(float);
    if (head_dim == 32) {
        flash_decoding_partial_kernel<InputT, kDecodeBlockN, 32><<<grid, kDecodeThreads, shared_partial, stream>>>(
            Q, K, V, partial_o, partial_m, partial_l, seq_len, num_chunks, chunk_len, scale);
    } else if (head_dim == 64) {
        flash_decoding_partial_kernel<InputT, kDecodeBlockN, 64><<<grid, kDecodeThreads, shared_partial, stream>>>(
            Q, K, V, partial_o, partial_m, partial_l, seq_len, num_chunks, chunk_len, scale);
    } else {
        flash_decoding_partial_kernel<InputT, kDecodeBlockN, 128><<<grid, kDecodeThreads, shared_partial, stream>>>(
            Q, K, V, partial_o, partial_m, partial_l, seq_len, num_chunks, chunk_len, scale);
    }

    dim3 combine_grid(bh_total);
    size_t shared_combine = (kDecodeThreads / 32) * sizeof(float);
    if (head_dim == 32) {
        flash_decoding_combine_kernel<InputT, 32><<<combine_grid, kDecodeThreads, shared_combine, stream>>>(
            partial_o, partial_m, partial_l, O, L, num_chunks, bh_total);
    } else if (head_dim == 64) {
        flash_decoding_combine_kernel<InputT, 64><<<combine_grid, kDecodeThreads, shared_combine, stream>>>(
            partial_o, partial_m, partial_l, O, L, num_chunks, bh_total);
    } else {
        flash_decoding_combine_kernel<InputT, 128><<<combine_grid, kDecodeThreads, shared_combine, stream>>>(
            partial_o, partial_m, partial_l, O, L, num_chunks, bh_total);
    }

    return FlashAttentionError::SUCCESS;
}

}  // namespace

FlashAttentionError flash_attention_decode(const float* Q, const float* K, const float* V, float* O,
                                           float* L, int batch_size, int num_heads, int seq_len,
                                           int head_dim, float scale, int num_chunks,
                                           cudaStream_t stream) {
    return launch_flash_decoding_typed(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim,
                                       scale, num_chunks, stream);
}

FlashAttentionError flash_attention_decode(const half* Q, const half* K, const half* V, half* O,
                                           float* L, int batch_size, int num_heads, int seq_len,
                                           int head_dim, float scale, int num_chunks,
                                           cudaStream_t stream) {
    return launch_flash_decoding_typed(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim,
                                       scale, num_chunks, stream);
}

FlashAttentionError flash_attention_decode(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                           const __nv_bfloat16* V, __nv_bfloat16* O, float* L,
                                           int batch_size, int num_heads, int seq_len, int head_dim,
                                           float scale, int num_chunks, cudaStream_t stream) {
    return launch_flash_decoding_typed(Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim,
                                       scale, num_chunks, stream);
}

}  // namespace cuflash
