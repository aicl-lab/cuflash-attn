// Tensor-core (WMMA) FlashAttention forward kernel for FP16/BF16.
//
// Phase 2 (forward) of docs/design/tensor-core-migration.md:
//   - Q/K/V tiles stay in the input precision in shared memory
//   - S = Q @ K^T and O += P @ V run on Tensor Cores (m16n16k16, FP32 accum)
//   - online softmax + FA2 deferred normalization are unchanged (CUDA cores)
// FP32 keeps the scalar path. BF16 fragments require sm_80+; FP16 requires
// sm_70+. On anything else the launcher falls back to the scalar kernel.

#include <float.h>
#include <mma.h>

#include <type_traits>

#include "cuflash/flash_attention.h"
#include "impl/tile_io.cuh"
#include "kernel_launch_utils.cuh"

namespace cuflash {

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_THREADS = 128;  // 4 warps, each warp owns 16 query rows

}  // namespace

template<typename InputT, int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void __launch_bounds__(WMMA_THREADS)
    flash_attention_forward_wmma_kernel(const InputT* __restrict__ Q, const InputT* __restrict__ K,
                                        const InputT* __restrict__ V, InputT* __restrict__ O,
                                        float* __restrict__ L, int seq_len, float scale,
                                        bool causal) {
#if defined(__CUDA_ARCH__)
    constexpr bool arch_ok =
        std::is_same_v<InputT, __nv_bfloat16> ? (__CUDA_ARCH__ >= 800) : (__CUDA_ARCH__ >= 700);
    if constexpr (arch_ok) {
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

        extern __shared__ __align__(16) char smem_raw[];
        InputT* Q_tile = reinterpret_cast<InputT*>(smem_raw);  // BLOCK_M x HEAD_DIM
        InputT* K_tile = Q_tile + BLOCK_M * HEAD_DIM;          // BLOCK_N x HEAD_DIM
        InputT* V_tile = K_tile + BLOCK_N * HEAD_DIM;          // BLOCK_N x HEAD_DIM
        float* S_tile = reinterpret_cast<float*>(V_tile + BLOCK_N * HEAD_DIM);  // BLOCK_M x BLOCK_N
        InputT* P_tile =
            reinterpret_cast<InputT*>(S_tile + BLOCK_M * BLOCK_N);             // BLOCK_M x BLOCK_N
        float* O_tile = reinterpret_cast<float*>(P_tile + BLOCK_M * BLOCK_N);  // BLOCK_M x HEAD_DIM
        float* m_tile = O_tile + BLOCK_M * HEAD_DIM;                           // BLOCK_M
        float* l_tile = m_tile + BLOCK_M;                                      // BLOCK_M

        const int tid = threadIdx.x;
        const int warp_row = (tid / 32) * WMMA_M;

        impl::load_tile_to_shared_native<BLOCK_M, HEAD_DIM>(Q_ptr, Q_tile, q_start, 0, seq_len,
                                                            HEAD_DIM, HEAD_DIM);
        for (int i = tid; i < BLOCK_M * HEAD_DIM; i += WMMA_THREADS) {
            O_tile[i] = 0.0f;
        }
        for (int i = tid; i < BLOCK_M; i += WMMA_THREADS) {
            m_tile[i] = -INFINITY;
            l_tile[i] = 0.0f;
        }
        __syncthreads();

        const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
        for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
            const int kv_start = kv_block * BLOCK_N;

            if (causal && kv_start > q_start + BLOCK_M - 1) {
                break;
            }

            impl::load_tile_to_shared_native<BLOCK_N, HEAD_DIM>(K_ptr, K_tile, kv_start, 0, seq_len,
                                                                HEAD_DIM, HEAD_DIM);
            impl::load_tile_to_shared_native<BLOCK_N, HEAD_DIM>(V_ptr, V_tile, kv_start, 0, seq_len,
                                                                HEAD_DIM, HEAD_DIM);
            __syncthreads();

            // S = scale * (Q @ K^T) on Tensor Cores; each warp owns a
            // 16 x BLOCK_N strip of S. K^T is realized by loading K with a
            // col_major fragment (no explicit transpose).
            if (warp_row < BLOCK_M) {
                for (int n0 = 0; n0 < BLOCK_N; n0 += WMMA_N) {
                    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
                        c_frag;
                    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
                    for (int k0 = 0; k0 < HEAD_DIM; k0 += WMMA_K) {
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                               InputT, nvcuda::wmma::row_major>
                            a_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                               InputT, nvcuda::wmma::col_major>
                            b_frag;
                        nvcuda::wmma::load_matrix_sync(a_frag, Q_tile + warp_row * HEAD_DIM + k0,
                                                       HEAD_DIM);
                        nvcuda::wmma::load_matrix_sync(b_frag, K_tile + n0 * HEAD_DIM + k0,
                                                       HEAD_DIM);
                        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
#pragma unroll
                    for (int i = 0; i < c_frag.num_elements; i++) {
                        c_frag.x[i] *= scale;
                    }
                    nvcuda::wmma::store_matrix_sync(S_tile + warp_row * BLOCK_N + n0, c_frag,
                                                    BLOCK_N, nvcuda::wmma::mem_row_major);
                }
            }
            __syncthreads();

            // Causal masking, online softmax, P = exp(S - m_new) conversion to
            // the input precision, and the O rescale, fused into one row-wise
            // pass (one thread per row).
            for (int row = tid; row < BLOCK_M; row += WMMA_THREADS) {
                if (q_start + row >= seq_len)
                    continue;

                float row_max = -INFINITY;
                for (int j = 0; j < BLOCK_N; j++) {
                    const bool valid =
                        (kv_start + j < seq_len) && (!causal || kv_start + j <= q_start + row);
                    if (valid) {
                        row_max = fmaxf(row_max, S_tile[row * BLOCK_N + j]);
                    }
                }

                const float m_old = m_tile[row];
                const float m_new = fmaxf(m_old, row_max);
                const float rescale = expf(m_old - m_new);

                for (int d = 0; d < HEAD_DIM; d++) {
                    O_tile[row * HEAD_DIM + d] *= rescale;
                }

                float row_sum = 0.0f;
                for (int j = 0; j < BLOCK_N; j++) {
                    const bool valid =
                        (kv_start + j < seq_len) && (!causal || kv_start + j <= q_start + row);
                    const float p = valid ? expf(S_tile[row * BLOCK_N + j] - m_new) : 0.0f;
                    P_tile[row * BLOCK_N + j] = static_cast<InputT>(p);
                    row_sum += p;
                }

                l_tile[row] = l_tile[row] * rescale + row_sum;
                m_tile[row] = m_new;
            }
            __syncthreads();

            // O += P @ V on Tensor Cores, accumulating in FP32. The previous
            // (rescaled) O is loaded as the accumulator fragment.
            if (warp_row < BLOCK_M) {
                for (int d0 = 0; d0 < HEAD_DIM; d0 += WMMA_N) {
                    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
                        c_frag;
                    nvcuda::wmma::load_matrix_sync(c_frag, O_tile + warp_row * HEAD_DIM + d0,
                                                   HEAD_DIM, nvcuda::wmma::mem_row_major);
                    for (int n0 = 0; n0 < BLOCK_N; n0 += WMMA_K) {
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                               InputT, nvcuda::wmma::row_major>
                            a_frag;
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                               InputT, nvcuda::wmma::row_major>
                            b_frag;
                        nvcuda::wmma::load_matrix_sync(a_frag, P_tile + warp_row * BLOCK_N + n0,
                                                       BLOCK_N);
                        nvcuda::wmma::load_matrix_sync(b_frag, V_tile + n0 * HEAD_DIM + d0,
                                                       HEAD_DIM);
                        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                    nvcuda::wmma::store_matrix_sync(O_tile + warp_row * HEAD_DIM + d0, c_frag,
                                                    HEAD_DIM, nvcuda::wmma::mem_row_major);
                }
            }
            __syncthreads();
        }

        // Deferred normalization: O = O_unnorm / l, L = m + log(l).
        for (int row = tid; row < BLOCK_M; row += WMMA_THREADS) {
            const int global_row = q_start + row;
            if (global_row >= seq_len)
                continue;

            const float l_inv = 1.0f / l_tile[row];
            for (int d = 0; d < HEAD_DIM; d++) {
                O_tile[row * HEAD_DIM + d] *= l_inv;
            }
            L_ptr[global_row] = m_tile[row] + logf(l_tile[row]);
        }
        __syncthreads();

        impl::store_tile_from_shared<BLOCK_M, HEAD_DIM>(O_tile, O_ptr, q_start, 0, seq_len,
                                                        HEAD_DIM, HEAD_DIM);
    }
#endif  // defined(__CUDA_ARCH__)
}

// Explicit kernel instantiations. Configs are sized so every layout fits the
// default 48 KB of dynamic shared memory (no opt-in needed):
//   hd32: (BM 64, BN 32) ~29 KB, hd64: (BM 64, BN 32) ~45 KB,
//   hd128: (BM 32, BN 32) ~47 KB.
template __global__ void flash_attention_forward_wmma_kernel<half, 64, 32, 32>(
    const half*, const half*, const half*, half*, float*, int, float, bool);
template __global__ void flash_attention_forward_wmma_kernel<half, 64, 32, 64>(
    const half*, const half*, const half*, half*, float*, int, float, bool);
template __global__ void flash_attention_forward_wmma_kernel<half, 32, 32, 128>(
    const half*, const half*, const half*, half*, float*, int, float, bool);

template __global__ void flash_attention_forward_wmma_kernel<__nv_bfloat16, 64, 32, 32>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    float, bool);
template __global__ void flash_attention_forward_wmma_kernel<__nv_bfloat16, 64, 32, 64>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    float, bool);
template __global__ void flash_attention_forward_wmma_kernel<__nv_bfloat16, 32, 32, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    float, bool);

namespace {

template<typename InputT>
constexpr size_t wmma_smem_bytes(int BM, int BN, int HD) {
    return static_cast<size_t>(BM * HD + 2 * BN * HD + BM * BN) * sizeof(InputT) +
           static_cast<size_t>(BM * BN + BM * HD + 2 * BM) * sizeof(float);
}

}  // namespace

template<typename InputT>
FlashAttentionError launch_flash_attention_forward_wmma_typed(
    const InputT* Q, const InputT* K, const InputT* V, InputT* O, float* L, int batch_size,
    int num_heads, int seq_len, int head_dim, float scale, bool causal, cudaStream_t stream) {
    const int batch_heads = batch_size * num_heads;
    const dim3 block(WMMA_THREADS);

    auto launch = [&](auto kernel_func, int BM, int BN, size_t smem_size) -> FlashAttentionError {
        // grid 展平到 x 维：gridDim.y 上限 65535，而 B*H 可能超过它。
        const int num_q_blocks = (seq_len + BM - 1) / BM;
        const dim3 grid(num_q_blocks * batch_heads);
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
        return launch(flash_attention_forward_wmma_kernel<InputT, 64, 32, 32>, 64, 32,
                      wmma_smem_bytes<InputT>(64, 32, 32));
    }
    if (head_dim == 64) {
        return launch(flash_attention_forward_wmma_kernel<InputT, 64, 32, 64>, 64, 32,
                      wmma_smem_bytes<InputT>(64, 32, 64));
    }
    return launch(flash_attention_forward_wmma_kernel<InputT, 32, 32, 128>, 32, 32,
                  wmma_smem_bytes<InputT>(32, 32, 128));
}

template FlashAttentionError launch_flash_attention_forward_wmma_typed<half>(
    const half*, const half*, const half*, half*, float*, int, int, int, int, float, bool,
    cudaStream_t);
template FlashAttentionError launch_flash_attention_forward_wmma_typed<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, int,
    int, int, int, float, bool, cudaStream_t);

}  // namespace cuflash
