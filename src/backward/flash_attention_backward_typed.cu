// Unified Flash Attention Backward Kernel - FP32/FP16/BF16 Template

#include <float.h>

#include "cuflash/flash_attention.h"
#include "impl/online_softmax.cuh"
#include "impl/tile_io.cuh"
#include "kernel_launch_utils.cuh"

namespace cuflash {

// Compute D = rowsum(dO * O) for each row
template<typename InputT, int BLOCK_SIZE, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    compute_D_kernel(const InputT* __restrict__ dO, const InputT* __restrict__ O,
                     float* __restrict__ D, int seq_len) {
    // grid 已展平到 x 维（total = d_blocks * batch_heads），避免
    // grid.y = B*H 在 B*H > 65535 时超出 CUDA 的 gridDim.y 上限。
    const int d_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int batch_head_idx = blockIdx.x / d_blocks;
    const int row_idx = (blockIdx.x % d_blocks) * BLOCK_SIZE + threadIdx.x;

    if (row_idx >= seq_len)
        return;

    const InputT* dO_row = dO + batch_head_idx * seq_len * HEAD_DIM + row_idx * HEAD_DIM;
    const InputT* O_row = O + batch_head_idx * seq_len * HEAD_DIM + row_idx * HEAD_DIM;

    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        sum += impl::TypeAdapter<InputT>::to_compute(dO_row[d]) *
               impl::TypeAdapter<InputT>::to_compute(O_row[d]);
    }

    D[batch_head_idx * seq_len + row_idx] = sum;
}

// Compute dQ for one q-block
template<typename InputT, int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_backward_dq_kernel(const InputT* __restrict__ Q, const InputT* __restrict__ K,
                                       const InputT* __restrict__ V, const float* __restrict__ L,
                                       const InputT* __restrict__ dO, const float* __restrict__ D,
                                       InputT* __restrict__ dQ, int seq_len, float scale,
                                       bool causal) {
    // grid 已展平到 x 维（total = num_q_blocks * batch_heads）。
    const int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int q_block_idx = blockIdx.x % num_q_blocks;
    const int batch_head_idx = blockIdx.x / num_q_blocks;

    const InputT* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    const float* L_ptr = L + batch_head_idx * seq_len;
    const InputT* dO_ptr = dO + batch_head_idx * seq_len * HEAD_DIM;
    const float* D_ptr = D + batch_head_idx * seq_len;
    InputT* dQ_ptr = dQ + batch_head_idx * seq_len * HEAD_DIM;

    const int q_start = q_block_idx * BLOCK_M;
    if (q_start >= seq_len)
        return;

    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float smem[];
    float* Q_tile = smem;                          // BLOCK_M x HEAD_DIM
    float* dO_tile = Q_tile + BLOCK_M * HEAD_DIM;  // BLOCK_M x HEAD_DIM
    float* K_tile = dO_tile + BLOCK_M * HEAD_DIM;  // BLOCK_N x HEAD_DIM
    float* V_tile = K_tile + BLOCK_N * HEAD_DIM;   // BLOCK_N x HEAD_DIM
    float* S_tile = V_tile + BLOCK_N * HEAD_DIM;   // BLOCK_M x BLOCK_N
    float* dQ_tile = S_tile + BLOCK_M * BLOCK_N;   // BLOCK_M x HEAD_DIM
    float* L_tile = dQ_tile + BLOCK_M * HEAD_DIM;  // BLOCK_M
    float* D_tile = L_tile + BLOCK_M;              // BLOCK_M

    impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(Q_ptr, Q_tile, q_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);
    impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(dO_ptr, dO_tile, q_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);

    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
        dQ_tile[i] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M; i += num_threads) {
        int global_idx = q_start + i;
        L_tile[i] = (global_idx < seq_len) ? L_ptr[global_idx] : 0.0f;
        D_tile[i] = (global_idx < seq_len) ? D_ptr[global_idx] : 0.0f;
    }
    __syncthreads();

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N;

        if (causal && kv_start > q_start + BLOCK_M - 1) {
            break;
        }

        impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(K_ptr, K_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);
        impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(V_ptr, V_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);
        __syncthreads();

        impl::matmul_ABt<BLOCK_M, BLOCK_N, HEAD_DIM>(Q_tile, K_tile, S_tile, scale);
        __syncthreads();

        // Compute dS = P * (dP - D)
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int q_idx = i / BLOCK_N;
            int k_idx = i % BLOCK_N;
            int global_q = q_start + q_idx;
            int global_k = kv_start + k_idx;

            if (global_q >= seq_len || global_k >= seq_len || (causal && global_k > global_q)) {
                S_tile[i] = 0.0f;
            } else {
                float dP = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dP += dO_tile[q_idx * HEAD_DIM + d] * V_tile[k_idx * HEAD_DIM + d];
                }
                float p = expf(S_tile[i] - L_tile[q_idx]);
                S_tile[i] = p * (dP - D_tile[q_idx]);
            }
        }
        __syncthreads();

        // dQ += dS^T @ K * scale (accumulate per-row)
        for (int q = tid; q < BLOCK_M; q += num_threads) {
            int global_q = q_start + q;
            if (global_q >= seq_len)
                continue;

            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int k = 0; k < BLOCK_N; k++) {
                    sum += S_tile[q * BLOCK_N + k] * K_tile[k * HEAD_DIM + d];
                }
                dQ_tile[q * HEAD_DIM + d] += sum * scale;
            }
        }
        __syncthreads();
    }

    impl::store_tile_from_shared<BLOCK_M, HEAD_DIM>(dQ_tile, dQ_ptr, q_start, 0, seq_len, HEAD_DIM,
                                                    HEAD_DIM);
}

// Compute dK and dV for one kv-block
template<typename InputT, int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_backward_dkdv_kernel(const InputT* __restrict__ Q, const InputT* __restrict__ K,
                                         const InputT* __restrict__ V, const float* __restrict__ L,
                                         const InputT* __restrict__ dO, const float* __restrict__ D,
                                         InputT* __restrict__ dK, InputT* __restrict__ dV,
                                         int seq_len, float scale, bool causal) {
    // grid 已展平到 x 维（total = num_kv_blocks * batch_heads）。
    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    const int kv_block_idx = blockIdx.x % num_kv_blocks;
    const int batch_head_idx = blockIdx.x / num_kv_blocks;

    const InputT* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    const float* L_ptr = L + batch_head_idx * seq_len;
    const InputT* dO_ptr = dO + batch_head_idx * seq_len * HEAD_DIM;
    const float* D_ptr = D + batch_head_idx * seq_len;
    InputT* dK_ptr = dK + batch_head_idx * seq_len * HEAD_DIM;
    InputT* dV_ptr = dV + batch_head_idx * seq_len * HEAD_DIM;

    const int kv_start = kv_block_idx * BLOCK_N;
    if (kv_start >= seq_len)
        return;

    const int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float smem[];
    float* K_tile = smem;                           // BLOCK_N x HEAD_DIM
    float* V_tile = K_tile + BLOCK_N * HEAD_DIM;    // BLOCK_N x HEAD_DIM
    float* Q_tile = V_tile + BLOCK_N * HEAD_DIM;    // BLOCK_M x HEAD_DIM
    float* dO_tile = Q_tile + BLOCK_M * HEAD_DIM;   // BLOCK_M x HEAD_DIM
    float* S_tile = dO_tile + BLOCK_M * HEAD_DIM;   // BLOCK_M x BLOCK_N
    float* dK_tile = S_tile + BLOCK_M * BLOCK_N;    // BLOCK_N x HEAD_DIM
    float* dV_tile = dK_tile + BLOCK_N * HEAD_DIM;  // BLOCK_N x HEAD_DIM
    float* L_tile = dV_tile + BLOCK_N * HEAD_DIM;   // BLOCK_M
    float* D_tile = L_tile + BLOCK_M;               // BLOCK_M

    impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(K_ptr, K_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);
    impl::load_tile_to_shared<BLOCK_N, HEAD_DIM>(V_ptr, V_tile, kv_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);

    for (int i = tid; i < BLOCK_N * HEAD_DIM; i += num_threads) {
        dK_tile[i] = 0.0f;
        dV_tile[i] = 0.0f;
    }
    __syncthreads();

    for (int q_block = 0; q_block < num_q_blocks; q_block++) {
        int q_start = q_block * BLOCK_M;

        if (causal && q_start + BLOCK_M - 1 < kv_start) {
            continue;
        }

        impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(Q_ptr, Q_tile, q_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);
        impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(dO_ptr, dO_tile, q_start, 0, seq_len, HEAD_DIM,
                                                     HEAD_DIM);

        for (int i = tid; i < BLOCK_M; i += num_threads) {
            int global_idx = q_start + i;
            L_tile[i] = (global_idx < seq_len) ? L_ptr[global_idx] : 0.0f;
            D_tile[i] = (global_idx < seq_len) ? D_ptr[global_idx] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T * scale, then P = softmax(S)
        impl::matmul_ABt<BLOCK_M, BLOCK_N, HEAD_DIM>(Q_tile, K_tile, S_tile, scale);
        __syncthreads();

        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int q_idx = i / BLOCK_N;
            int k_idx = i % BLOCK_N;
            int global_q = q_start + q_idx;
            int global_k = kv_start + k_idx;

            if (global_q >= seq_len || global_k >= seq_len || (causal && global_k > global_q)) {
                S_tile[i] = 0.0f;
            } else {
                S_tile[i] = expf(S_tile[i] - L_tile[q_idx]);
            }
        }
        __syncthreads();

        // dV += P^T @ dO
        for (int k = tid; k < BLOCK_N; k += num_threads) {
            if (kv_start + k >= seq_len)
                continue;
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int q = 0; q < BLOCK_M; q++) {
                    if (q_start + q < seq_len) {
                        sum += S_tile[q * BLOCK_N + k] * dO_tile[q * HEAD_DIM + d];
                    }
                }
                dV_tile[k * HEAD_DIM + d] += sum;
            }
        }
        __syncthreads();

        // dS = P * (dP - D)
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int q_idx = i / BLOCK_N;
            int k_idx = i % BLOCK_N;
            int global_q = q_start + q_idx;
            int global_k = kv_start + k_idx;

            if (global_q >= seq_len || global_k >= seq_len || (causal && global_k > global_q)) {
                S_tile[i] = 0.0f;
            } else {
                float dP = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dP += dO_tile[q_idx * HEAD_DIM + d] * V_tile[k_idx * HEAD_DIM + d];
                }
                S_tile[i] = S_tile[i] * (dP - D_tile[q_idx]);
            }
        }
        __syncthreads();

        // dK += dS^T @ Q * scale
        for (int k = tid; k < BLOCK_N; k += num_threads) {
            if (kv_start + k >= seq_len)
                continue;
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int q = 0; q < BLOCK_M; q++) {
                    if (q_start + q < seq_len) {
                        sum += S_tile[q * BLOCK_N + k] * Q_tile[q * HEAD_DIM + d];
                    }
                }
                dK_tile[k * HEAD_DIM + d] += sum * scale;
            }
        }
        __syncthreads();
    }

    impl::store_tile_from_shared<BLOCK_N, HEAD_DIM>(dK_tile, dK_ptr, kv_start, 0, seq_len, HEAD_DIM,
                                                    HEAD_DIM);
    impl::store_tile_from_shared<BLOCK_N, HEAD_DIM>(dV_tile, dV_ptr, kv_start, 0, seq_len, HEAD_DIM,
                                                    HEAD_DIM);
}

// Explicit template instantiations for FP32
template __global__ void compute_D_kernel<float, 128, 32>(const float*, const float*, float*, int);
template __global__ void compute_D_kernel<float, 128, 64>(const float*, const float*, float*, int);
template __global__ void compute_D_kernel<float, 128, 128>(const float*, const float*, float*, int);

template __global__ void flash_attention_backward_dq_kernel<float, 64, 64, 32>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<float, 32, 32, 64>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<float, 16, 32, 128>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*, int,
    float, bool);

template __global__ void flash_attention_backward_dkdv_kernel<float, 64, 64, 32>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<float, 32, 32, 64>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<float, 16, 32, 128>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, int, float, bool);

// Explicit template instantiations for FP16
template __global__ void compute_D_kernel<half, 128, 32>(const half*, const half*, float*, int);
template __global__ void compute_D_kernel<half, 128, 64>(const half*, const half*, float*, int);
template __global__ void compute_D_kernel<half, 128, 128>(const half*, const half*, float*, int);

template __global__ void flash_attention_backward_dq_kernel<half, 64, 64, 32>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<half, 32, 32, 64>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<half, 16, 32, 128>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, int,
    float, bool);

template __global__ void flash_attention_backward_dkdv_kernel<half, 64, 64, 32>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, half*,
    int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<half, 32, 32, 64>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, half*,
    int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<half, 16, 32, 128>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, half*,
    int, float, bool);

// Explicit template instantiations for BF16
template __global__ void compute_D_kernel<__nv_bfloat16, 128, 32>(const __nv_bfloat16*,
                                                                  const __nv_bfloat16*, float*,
                                                                  int);
template __global__ void compute_D_kernel<__nv_bfloat16, 128, 64>(const __nv_bfloat16*,
                                                                  const __nv_bfloat16*, float*,
                                                                  int);
template __global__ void compute_D_kernel<__nv_bfloat16, 128, 128>(const __nv_bfloat16*,
                                                                   const __nv_bfloat16*, float*,
                                                                   int);

template __global__ void flash_attention_backward_dq_kernel<__nv_bfloat16, 64, 64, 32>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, int, float, bool);
template __global__ void flash_attention_backward_dq_kernel<__nv_bfloat16, 32, 32, 64>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, int, float, bool);
template __global__ void flash_attention_backward_dq_kernel<__nv_bfloat16, 16, 32, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, int, float, bool);

template __global__ void flash_attention_backward_dkdv_kernel<__nv_bfloat16, 64, 64, 32>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, __nv_bfloat16*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<__nv_bfloat16, 32, 32, 64>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, __nv_bfloat16*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<__nv_bfloat16, 16, 32, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, __nv_bfloat16*, int, float, bool);

namespace {

// RAII device buffer freed with cudaFreeAsync so the free is stream-ordered.
struct AsyncFloatBuffer {
    float* ptr = nullptr;
    cudaStream_t stream = 0;

    FlashAttentionError alloc(size_t elements, cudaStream_t s) {
        stream = s;
        cudaError_t err = cudaMallocAsync(&ptr, elements * sizeof(float), s);
        if (err != cudaSuccess) {
            ptr = nullptr;
            return err == cudaErrorMemoryAllocation ? FlashAttentionError::OUT_OF_MEMORY
                                                    : FlashAttentionError::CUDA_ERROR;
        }
        return FlashAttentionError::SUCCESS;
    }

    ~AsyncFloatBuffer() {
        if (ptr != nullptr) {
            cudaFreeAsync(ptr, stream);
        }
    }
};

}  // namespace

// Fallback tilings for GPUs with a small shared-memory cap (e.g. sm_75).

template __global__ void flash_attention_backward_dq_kernel<float, 32, 32, 32>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<float, 16, 16, 128>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*, int,
    float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<float, 32, 32, 32>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<float, 16, 16, 128>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, int, float, bool);

template __global__ void flash_attention_backward_dq_kernel<half, 32, 32, 32>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dq_kernel<half, 16, 16, 128>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, int,
    float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<half, 32, 32, 32>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, half*,
    int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<half, 16, 16, 128>(
    const half*, const half*, const half*, const float*, const half*, const float*, half*, half*,
    int, float, bool);

template __global__ void flash_attention_backward_dq_kernel<__nv_bfloat16, 32, 32, 32>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, int, float, bool);
template __global__ void flash_attention_backward_dq_kernel<__nv_bfloat16, 16, 16, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<__nv_bfloat16, 32, 32, 32>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, __nv_bfloat16*, int, float, bool);
template __global__ void flash_attention_backward_dkdv_kernel<__nv_bfloat16, 16, 16, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const float*, __nv_bfloat16*, __nv_bfloat16*, int, float, bool);

// Unified launch function - single generic implementation for all dtypes
template<typename InputT>
FlashAttentionError launch_flash_attention_backward_typed(
    const InputT* Q, const InputT* K, const InputT* V, const InputT* O, const float* L,
    const InputT* dO, InputT* dQ, InputT* dK, InputT* dV, int batch_size, int num_heads,
    int seq_len, int head_dim, float scale, bool causal, cudaStream_t stream) {
    using Config = impl::BackwardTilingConfig;

    int batch_heads = batch_size * num_heads;

    // Stream-ordered allocation: each call gets its own D buffer tied to the
    // caller's stream, so concurrent backward calls on different streams never
    // share one buffer, and the memory is only reused after every kernel
    // queued before the free on this stream has completed.
    AsyncFloatBuffer d_buffer;
    FlashAttentionError workspace_status =
        d_buffer.alloc(static_cast<size_t>(batch_heads) * static_cast<size_t>(seq_len), stream);
    if (workspace_status != FlashAttentionError::SUCCESS) {
        return workspace_status;
    }
    float* D = d_buffer.ptr;

    // Phase 1: compute D = rowsum(dO * O)
    int d_blocks = (seq_len + 127) / 128;
    // grid 展平到 x 维：gridDim.y 上限 65535，而 B*H 可能超过它。
    dim3 d_grid(d_blocks * batch_heads);

    if (head_dim == 32) {
        compute_D_kernel<InputT, 128, 32><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else if (head_dim == 64) {
        compute_D_kernel<InputT, 128, 64><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    } else {
        compute_D_kernel<InputT, 128, 128><<<d_grid, 128, 0, stream>>>(dO, O, D, seq_len);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FlashAttentionError::CUDA_ERROR;
    }

    // Phase 2: compute dQ and dK/dV
    int max_dynamic_smem = 0;
    FlashAttentionError status = query_max_dynamic_shared_memory_per_block(&max_dynamic_smem);
    if (status != FlashAttentionError::SUCCESS) {
        return status;
    }

    // Pick the largest tiling whose shared memory (both kernels) fits this
    // device, falling back to smaller tiles on low-cap GPUs (sm_75: 64 KB).
    int BM, BN;
    if (head_dim == 32) {
        BM = Config::BLOCK_M;
        BN = Config::BLOCK_N;
        const size_t dq_need = Config::dq_smem_bytes(head_dim, BM, BN);
        const size_t dkdv_need = Config::dkdv_smem_bytes(head_dim, BM, BN);
        const size_t need = dq_need > dkdv_need ? dq_need : dkdv_need;
        if (need > static_cast<size_t>(max_dynamic_smem)) {
            BM = Config::BLOCK_M_SMALL;
            BN = Config::BLOCK_N_SMALL;
        }
    } else if (head_dim == 64) {
        BM = Config::BLOCK_M_HD64;
        BN = Config::BLOCK_N_HD64;
    } else {
        BM = Config::BLOCK_M_HD128;
        BN = Config::BLOCK_N_HD128;
        const size_t dq_need = Config::dq_smem_bytes(head_dim, BM, BN);
        const size_t dkdv_need = Config::dkdv_smem_bytes(head_dim, BM, BN);
        const size_t need = dq_need > dkdv_need ? dq_need : dkdv_need;
        if (need > static_cast<size_t>(max_dynamic_smem)) {
            BN = Config::BLOCK_N_HD128_SMALL;
        }
    }

    const dim3 block(Config::NUM_THREADS);
    const dim3 dq_grid(((seq_len + BM - 1) / BM) * batch_heads);
    const dim3 dkdv_grid(((seq_len + BN - 1) / BN) * batch_heads);
    const size_t dq_smem = Config::dq_smem_bytes(head_dim, BM, BN);
    const size_t dkdv_smem = Config::dkdv_smem_bytes(head_dim, BM, BN);

    auto launch_dq = [&](auto kernel_func) -> FlashAttentionError {
        FlashAttentionError prep =
            prepare_dynamic_smem_launch(reinterpret_cast<const void*>(kernel_func), dq_smem);
        if (prep != FlashAttentionError::SUCCESS) {
            return prep;
        }
        kernel_func<<<dq_grid, block, dq_smem, stream>>>(Q, K, V, L, dO, D, dQ, seq_len, scale,
                                                         causal);
        return cudaGetLastError() == cudaSuccess ? FlashAttentionError::SUCCESS
                                                 : FlashAttentionError::CUDA_ERROR;
    };
    auto launch_dkdv = [&](auto kernel_func) -> FlashAttentionError {
        FlashAttentionError prep =
            prepare_dynamic_smem_launch(reinterpret_cast<const void*>(kernel_func), dkdv_smem);
        if (prep != FlashAttentionError::SUCCESS) {
            return prep;
        }
        kernel_func<<<dkdv_grid, block, dkdv_smem, stream>>>(Q, K, V, L, dO, D, dK, dV, seq_len,
                                                             scale, causal);
        return cudaGetLastError() == cudaSuccess ? FlashAttentionError::SUCCESS
                                                 : FlashAttentionError::CUDA_ERROR;
    };

    if (head_dim == 32) {
        if (BM == Config::BLOCK_M) {
            status = launch_dq(
                flash_attention_backward_dq_kernel<InputT, Config::BLOCK_M, Config::BLOCK_N, 32>);
            if (status != FlashAttentionError::SUCCESS)
                return status;
            return launch_dkdv(
                flash_attention_backward_dkdv_kernel<InputT, Config::BLOCK_M, Config::BLOCK_N, 32>);
        }
        status = launch_dq(flash_attention_backward_dq_kernel<InputT, Config::BLOCK_M_SMALL,
                                                              Config::BLOCK_N_SMALL, 32>);
        if (status != FlashAttentionError::SUCCESS)
            return status;
        return launch_dkdv(flash_attention_backward_dkdv_kernel<InputT, Config::BLOCK_M_SMALL,
                                                                Config::BLOCK_N_SMALL, 32>);
    }
    if (head_dim == 64) {
        status = launch_dq(flash_attention_backward_dq_kernel<InputT, Config::BLOCK_M_HD64,
                                                              Config::BLOCK_N_HD64, 64>);
        if (status != FlashAttentionError::SUCCESS)
            return status;
        return launch_dkdv(flash_attention_backward_dkdv_kernel<InputT, Config::BLOCK_M_HD64,
                                                                Config::BLOCK_N_HD64, 64>);
    }
    if (BN == Config::BLOCK_N_HD128) {
        status = launch_dq(flash_attention_backward_dq_kernel<InputT, Config::BLOCK_M_HD128,
                                                              Config::BLOCK_N_HD128, 128>);
        if (status != FlashAttentionError::SUCCESS)
            return status;
        return launch_dkdv(flash_attention_backward_dkdv_kernel<InputT, Config::BLOCK_M_HD128,
                                                                Config::BLOCK_N_HD128, 128>);
    }
    status = launch_dq(flash_attention_backward_dq_kernel<InputT, Config::BLOCK_M_HD128,
                                                          Config::BLOCK_N_HD128_SMALL, 128>);
    if (status != FlashAttentionError::SUCCESS)
        return status;
    return launch_dkdv(flash_attention_backward_dkdv_kernel<InputT, Config::BLOCK_M_HD128,
                                                            Config::BLOCK_N_HD128_SMALL, 128>);
}
// Explicit instantiations for supported dtypes
template FlashAttentionError launch_flash_attention_backward_typed<float>(
    const float*, const float*, const float*, const float*, const float*, const float*, float*,
    float*, float*, int, int, int, int, float, bool, cudaStream_t);
template FlashAttentionError launch_flash_attention_backward_typed<half>(
    const half*, const half*, const half*, const half*, const float*, const half*, half*, half*,
    half*, int, int, int, int, float, bool, cudaStream_t);
template FlashAttentionError launch_flash_attention_backward_typed<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const float*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int,
    int, int, float, bool, cudaStream_t);

}  // namespace cuflash
