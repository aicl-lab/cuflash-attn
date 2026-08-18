// Unified Flash Attention Forward Kernel - FP32/FP16/BF16 Template

#include <float.h>

#include <type_traits>

#include "cuflash/flash_attention.h"
#include "impl/online_softmax.cuh"
#include "impl/tile_io.cuh"
#include "kernel_launch_utils.cuh"

namespace cuflash {

// FA2-style forward kernel with deferred normalization.
// Maintains unnormalized O; final division by l happens once at the end.
template<typename InputT, int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(128)
    flash_attention_forward_kernel(const InputT* __restrict__ Q, const InputT* __restrict__ K,
                                   const InputT* __restrict__ V, InputT* __restrict__ O,
                                   float* __restrict__ L, int seq_len, float scale, bool causal) {
    // grid 已展平到 x 维（total = num_q_blocks * batch_heads），避免
    // grid.y = B*H 在 B*H > 65535 时超出 CUDA 的 gridDim.y 上限。
    const int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    const int q_block_idx = blockIdx.x % num_q_blocks;
    const int batch_head_idx = blockIdx.x / num_q_blocks;

    const InputT* Q_ptr = Q + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* K_ptr = K + batch_head_idx * seq_len * HEAD_DIM;
    const InputT* V_ptr = V + batch_head_idx * seq_len * HEAD_DIM;
    InputT* O_ptr = O + batch_head_idx * seq_len * HEAD_DIM;
    float* L_ptr = L + batch_head_idx * seq_len;

    const int q_start = q_block_idx * BLOCK_M;
    if (q_start >= seq_len)
        return;

    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;

    extern __shared__ float smem[];
    float* Q_tile = smem;                         // BLOCK_M x HEAD_DIM
    float* K_tile = Q_tile + BLOCK_M * HEAD_DIM;  // BLOCK_N x HEAD_DIM
    float* V_tile = K_tile + BLOCK_N * HEAD_DIM;  // BLOCK_N x HEAD_DIM
    float* S_tile = V_tile + BLOCK_N * HEAD_DIM;  // BLOCK_M x BLOCK_N
    float* O_tile = S_tile + BLOCK_M * BLOCK_N;   // BLOCK_M x HEAD_DIM
    float* m_tile = O_tile + BLOCK_M * HEAD_DIM;  // BLOCK_M
    float* l_tile = m_tile + BLOCK_M;             // BLOCK_M

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    impl::load_tile_to_shared<BLOCK_M, HEAD_DIM>(Q_ptr, Q_tile, q_start, 0, seq_len, HEAD_DIM,
                                                 HEAD_DIM);

    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += num_threads) {
        O_tile[i] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M; i += num_threads) {
        m_tile[i] = -INFINITY;
        l_tile[i] = 0.0f;
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

        // S = Q @ K^T * scale
        impl::matmul_ABt<BLOCK_M, BLOCK_N, HEAD_DIM>(Q_tile, K_tile, S_tile, scale);
        __syncthreads();

        if (causal) {
            for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
                int q_idx = i / BLOCK_N;
                int k_idx = i % BLOCK_N;
                if (kv_start + k_idx > q_start + q_idx) {
                    S_tile[i] = -INFINITY;
                }
            }
            __syncthreads();
        }

        // Online softmax update + FA2 deferred normalization:
        // P[j] = exp(S[j] - m_new), O += P @ V (no per-iteration 1/l division)
        for (int row = tid; row < BLOCK_M; row += num_threads) {
            if (q_start + row >= seq_len)
                continue;

            float row_max = -INFINITY;
            for (int j = 0; j < BLOCK_N; j++) {
                if (kv_start + j < seq_len) {
                    row_max = fmaxf(row_max, S_tile[row * BLOCK_N + j]);
                }
            }

            float m_old = m_tile[row];
            float m_new = fmaxf(m_old, row_max);
            float rescale = expf(m_old - m_new);

            // Rescale existing O accumulator
            for (int d = 0; d < HEAD_DIM; d++) {
                O_tile[row * HEAD_DIM + d] *= rescale;
            }

            // P = exp(S - m_new), computed once per element and kept in
            // S_tile for the P @ V product below (out-of-range entries are 0).
            float row_sum = 0.0f;
            for (int j = 0; j < BLOCK_N; j++) {
                float p = 0.0f;
                if (kv_start + j < seq_len) {
                    p = expf(S_tile[row * BLOCK_N + j] - m_new);
                }
                S_tile[row * BLOCK_N + j] = p;
                row_sum += p;
            }

            // O += P @ V
            for (int d = 0; d < HEAD_DIM; d++) {
                float pv = 0.0f;
                for (int j = 0; j < BLOCK_N; j++) {
                    pv += S_tile[row * BLOCK_N + j] * V_tile[j * HEAD_DIM + d];
                }
                O_tile[row * HEAD_DIM + d] += pv;
            }

            // Update running statistics
            l_tile[row] = l_tile[row] * rescale + row_sum;
            m_tile[row] = m_new;
        }
        __syncthreads();
    }

    // Final normalization: O = O_unnorm / l
    for (int row = tid; row < BLOCK_M; row += num_threads) {
        int global_row = q_start + row;
        if (global_row >= seq_len)
            continue;

        float l_inv = 1.0f / l_tile[row];
        for (int d = 0; d < HEAD_DIM; d++) {
            O_ptr[global_row * HEAD_DIM + d] =
                impl::TypeAdapter<InputT>::from_compute(O_tile[row * HEAD_DIM + d] * l_inv);
        }

        L_ptr[global_row] = m_tile[row] + logf(l_tile[row]);
    }
}

// Explicit template instantiations
template __global__ void flash_attention_forward_kernel<float, 64, 64, 32>(
    const float*, const float*, const float*, float*, float*, int, float, bool);
template __global__ void flash_attention_forward_kernel<float, 64, 64, 64>(
    const float*, const float*, const float*, float*, float*, int, float, bool);
template __global__ void flash_attention_forward_kernel<float, 32, 32, 128>(
    const float*, const float*, const float*, float*, float*, int, float, bool);

template __global__ void flash_attention_forward_kernel<half, 64, 64, 32>(const half*, const half*,
                                                                          const half*, half*,
                                                                          float*, int, float, bool);
template __global__ void flash_attention_forward_kernel<half, 64, 64, 64>(const half*, const half*,
                                                                          const half*, half*,
                                                                          float*, int, float, bool);
template __global__ void flash_attention_forward_kernel<half, 32, 32, 128>(const half*, const half*,
                                                                           const half*, half*,
                                                                           float*, int, float,
                                                                           bool);

template __global__ void flash_attention_forward_kernel<__nv_bfloat16, 64, 64, 32>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    float, bool);
template __global__ void flash_attention_forward_kernel<__nv_bfloat16, 64, 64, 64>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    float, bool);
template __global__ void flash_attention_forward_kernel<__nv_bfloat16, 32, 32, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    float, bool);

// Fallback tilings for GPUs with a small shared-memory cap (e.g. sm_75).
template __global__ void flash_attention_forward_kernel<float, 32, 32, 64>(
    const float*, const float*, const float*, float*, float*, int, float, bool);
template __global__ void flash_attention_forward_kernel<float, 16, 16, 128>(
    const float*, const float*, const float*, float*, float*, int, float, bool);

template __global__ void flash_attention_forward_kernel<half, 32, 32, 64>(const half*, const half*,
                                                                          const half*, half*,
                                                                          float*, int, float, bool);
template __global__ void flash_attention_forward_kernel<half, 16, 16, 128>(const half*, const half*,
                                                                           const half*, half*,
                                                                           float*, int, float,
                                                                           bool);

template __global__ void flash_attention_forward_kernel<__nv_bfloat16, 32, 32, 64>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    float, bool);
template __global__ void flash_attention_forward_kernel<__nv_bfloat16, 16, 16, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    float, bool);

// Tensor-core forward (Phase 2 of the tensor-core migration plan), defined
// in flash_attention_forward_wmma.cu.
template<typename InputT>
FlashAttentionError launch_flash_attention_forward_wmma_typed(
    const InputT* Q, const InputT* K, const InputT* V, InputT* O, float* L, int batch_size,
    int num_heads, int seq_len, int head_dim, float scale, bool causal, cudaStream_t stream);

// Unified launch function - single generic implementation for all dtypes
template<typename InputT>
FlashAttentionError launch_flash_attention_forward_typed(const InputT* Q, const InputT* K,
                                                         const InputT* V, InputT* O, float* L,
                                                         int batch_size, int num_heads, int seq_len,
                                                         int head_dim, float scale, bool causal,
                                                         cudaStream_t stream) {
    using Config = impl::ForwardTilingConfig;

    // Tensor-core path for reduced precision. BF16 fragments need sm_80+,
    // FP16 needs sm_70+; anything else keeps the scalar path below. A
    // CUDA_ERROR from the WMMA launch (e.g. no binary compiled for this
    // arch) also falls through to the scalar path.
    if constexpr (!std::is_same_v<InputT, float>) {
        int device = 0;
        int major = 0;
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) ==
                cudaSuccess) {
            const bool needs_sm80 = std::is_same_v<InputT, __nv_bfloat16>;
            if (major >= 8 || (major == 7 && !needs_sm80)) {
                FlashAttentionError wmma_status = launch_flash_attention_forward_wmma_typed<InputT>(
                    Q, K, V, O, L, batch_size, num_heads, seq_len, head_dim, scale, causal, stream);
                if (wmma_status != FlashAttentionError::CUDA_ERROR) {
                    return wmma_status;
                }
            }
        }
    }

    int max_dynamic_smem = 0;
    FlashAttentionError status = query_max_dynamic_shared_memory_per_block(&max_dynamic_smem);
    if (status != FlashAttentionError::SUCCESS) {
        return status;
    }

    // Pick the largest tiling whose shared memory fits this device, falling
    // back to smaller tiles on GPUs with a low cap (sm_75 opt-in is 64 KB).
    int BM, BN;
    if (head_dim == 32) {
        BM = Config::BLOCK_M;
        BN = Config::BLOCK_N;
    } else if (head_dim == 64) {
        BM = Config::BLOCK_M;
        BN = Config::BLOCK_N;
        if (Config::smem_bytes(head_dim, BM, BN) > static_cast<size_t>(max_dynamic_smem)) {
            BM = Config::BLOCK_M_SMALL;
            BN = Config::BLOCK_N_SMALL;
        }
    } else {
        BM = Config::BLOCK_M_HD128;
        BN = Config::BLOCK_N_HD128;
        if (Config::smem_bytes(head_dim, BM, BN) > static_cast<size_t>(max_dynamic_smem)) {
            BM = Config::BLOCK_M_HD128_SMALL;
            BN = Config::BLOCK_N_HD128_SMALL;
        }
    }

    const int batch_heads = batch_size * num_heads;
    // grid 展平到 x 维：gridDim.y 上限 65535，而 B*H 可能超过它。
    const int num_q_blocks = (seq_len + BM - 1) / BM;
    const dim3 grid(num_q_blocks * batch_heads);
    const dim3 block(Config::NUM_THREADS);
    const size_t smem_size = Config::smem_bytes(head_dim, BM, BN);

    auto launch = [&](auto kernel_func) -> FlashAttentionError {
        FlashAttentionError prep =
            prepare_dynamic_smem_launch(reinterpret_cast<const void*>(kernel_func), smem_size);
        if (prep != FlashAttentionError::SUCCESS) {
            return prep;
        }
        kernel_func<<<grid, block, smem_size, stream>>>(Q, K, V, O, L, seq_len, scale, causal);
        return cudaGetLastError() == cudaSuccess ? FlashAttentionError::SUCCESS
                                                 : FlashAttentionError::CUDA_ERROR;
    };

    if (head_dim == 32) {
        return launch(flash_attention_forward_kernel<InputT, Config::BLOCK_M, Config::BLOCK_N, 32>);
    }
    if (head_dim == 64) {
        if (BM == Config::BLOCK_M) {
            return launch(
                flash_attention_forward_kernel<InputT, Config::BLOCK_M, Config::BLOCK_N, 64>);
        }
        return launch(flash_attention_forward_kernel<InputT, Config::BLOCK_M_SMALL,
                                                     Config::BLOCK_N_SMALL, 64>);
    }
    if (BM == Config::BLOCK_M_HD128) {
        return launch(flash_attention_forward_kernel<InputT, Config::BLOCK_M_HD128,
                                                     Config::BLOCK_N_HD128, 128>);
    }
    return launch(flash_attention_forward_kernel<InputT, Config::BLOCK_M_HD128_SMALL,
                                                 Config::BLOCK_N_HD128_SMALL, 128>);
}

// Explicit instantiations for supported dtypes
template FlashAttentionError launch_flash_attention_forward_typed<float>(const float*, const float*,
                                                                         const float*, float*,
                                                                         float*, int, int, int, int,
                                                                         float, bool, cudaStream_t);
template FlashAttentionError launch_flash_attention_forward_typed<half>(const half*, const half*,
                                                                        const half*, half*, float*,
                                                                        int, int, int, int, float,
                                                                        bool, cudaStream_t);
template FlashAttentionError launch_flash_attention_forward_typed<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    int, int, int, float, bool, cudaStream_t);

}  // namespace cuflash
