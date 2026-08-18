#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuflash/export.h"

namespace cuflash {

// Error codes
enum class FlashAttentionError {
    SUCCESS = 0,
    INVALID_DIMENSION,     // 维度参数无效
    DIMENSION_MISMATCH,    // Q, K, V 张量形状不匹配
    NULL_POINTER,          // 空指针输入
    CUDA_ERROR,            // CUDA 运行时错误
    OUT_OF_MEMORY,         // 显存不足
    UNSUPPORTED_HEAD_DIM,  // 不支持的 head_dim
    UNSUPPORTED_DTYPE      // 不支持的数据类型
};

// Get error message string
CUFLASH_EXPORT const char* get_error_string(FlashAttentionError error);

// Forward pass
// Q, K, V: [batch_size, num_heads, seq_len, head_dim]
// O: [batch_size, num_heads, seq_len, head_dim]
// L: [batch_size, num_heads, seq_len] - logsumexp for backward
CUFLASH_EXPORT FlashAttentionError flash_attention_forward(const float* Q, const float* K,
                                                           const float* V, float* O, float* L,
                                                           int batch_size, int num_heads,
                                                           int seq_len, int head_dim, float scale,
                                                           bool causal, cudaStream_t stream = 0);

// Backward pass
CUFLASH_EXPORT FlashAttentionError flash_attention_backward(
    const float* Q, const float* K, const float* V, const float* O, const float* L, const float* dO,
    float* dQ, float* dK, float* dV, int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream = 0);

// Half precision versions.
// NOTE: L (logsumexp) is always float, even for reduced-precision inputs. The
// backward pass reconstructs softmax probabilities as exp(S - L); storing L in
// half would round the normalization constant and systematically corrupt the
// gradients. This matches the reference FlashAttention, which keeps softmax_lse
// in FP32.
CUFLASH_EXPORT FlashAttentionError flash_attention_forward(const half* Q, const half* K,
                                                           const half* V, half* O, float* L,
                                                           int batch_size, int num_heads,
                                                           int seq_len, int head_dim, float scale,
                                                           bool causal, cudaStream_t stream = 0);

CUFLASH_EXPORT FlashAttentionError flash_attention_backward(
    const half* Q, const half* K, const half* V, const half* O, const float* L, const half* dO,
    half* dQ, half* dK, half* dV, int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream = 0);

// BFloat16 precision versions (L is float for the same reason as above).
CUFLASH_EXPORT FlashAttentionError flash_attention_forward(const __nv_bfloat16* Q,
                                                           const __nv_bfloat16* K,
                                                           const __nv_bfloat16* V, __nv_bfloat16* O,
                                                           float* L, int batch_size, int num_heads,
                                                           int seq_len, int head_dim, float scale,
                                                           bool causal, cudaStream_t stream = 0);

CUFLASH_EXPORT FlashAttentionError flash_attention_backward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, const __nv_bfloat16* O,
    const float* L, const __nv_bfloat16* dO, __nv_bfloat16* dQ, __nv_bfloat16* dK,
    __nv_bfloat16* dV, int batch_size, int num_heads, int seq_len, int head_dim, float scale,
    bool causal, cudaStream_t stream = 0);

// Decode pass (FlashDecoding / Split-KV) — query_len == 1 against a long KV.
// Q: [batch_size * num_heads, 1, head_dim]
// K/V: [batch_size * num_heads, seq_len, head_dim]
// O: [batch_size * num_heads, 1, head_dim]
// L: [batch_size * num_heads]  (logsumexp, for downstream use/backward)
// num_chunks: 沿 KV 序列维拆分的块数（>0）；会被 clamp 到合法上限。
//   每块独立计算局部 online softmax，再跨块归约（Split-KV/FlashDecoding）。
// 语义：单 query 对全部 KV 行做 attention（decode 阶段 causal 的
// query 位于 seq_len-1，注意力集合即 [0, seq_len)）。
CUFLASH_EXPORT FlashAttentionError flash_attention_decode(const float* Q, const float* K,
                                                          const float* V, float* O, float* L,
                                                          int batch_size, int num_heads,
                                                          int seq_len, int head_dim, float scale,
                                                          int num_chunks, cudaStream_t stream = 0);

CUFLASH_EXPORT FlashAttentionError flash_attention_decode(const half* Q, const half* K,
                                                          const half* V, half* O, float* L,
                                                          int batch_size, int num_heads,
                                                          int seq_len, int head_dim, float scale,
                                                          int num_chunks, cudaStream_t stream = 0);

CUFLASH_EXPORT FlashAttentionError flash_attention_decode(const __nv_bfloat16* Q,
                                                          const __nv_bfloat16* K,
                                                          const __nv_bfloat16* V, __nv_bfloat16* O,
                                                          float* L, int batch_size, int num_heads,
                                                          int seq_len, int head_dim, float scale,
                                                          int num_chunks, cudaStream_t stream = 0);

}  // namespace cuflash

#ifdef __cplusplus
extern "C" {
#endif

// Stable C ABI wrappers for shared-library integration.
// Return values are the integer representation of cuflash::FlashAttentionError.
CUFLASH_EXPORT int cuflash_attention_forward_f32(const float* Q, const float* K, const float* V,
                                                 float* O, float* L, int batch_size, int num_heads,
                                                 int seq_len, int head_dim, float scale,
                                                 bool causal, cudaStream_t stream);

CUFLASH_EXPORT int cuflash_attention_backward_f32(const float* Q, const float* K, const float* V,
                                                  const float* O, const float* L, const float* dO,
                                                  float* dQ, float* dK, float* dV, int batch_size,
                                                  int num_heads, int seq_len, int head_dim,
                                                  float scale, bool causal, cudaStream_t stream);

CUFLASH_EXPORT int cuflash_attention_forward_f16(const half* Q, const half* K, const half* V,
                                                 half* O, float* L, int batch_size, int num_heads,
                                                 int seq_len, int head_dim, float scale,
                                                 bool causal, cudaStream_t stream);

CUFLASH_EXPORT int cuflash_attention_backward_f16(const half* Q, const half* K, const half* V,
                                                  const half* O, const float* L, const half* dO,
                                                  half* dQ, half* dK, half* dV, int batch_size,
                                                  int num_heads, int seq_len, int head_dim,
                                                  float scale, bool causal, cudaStream_t stream);

CUFLASH_EXPORT int cuflash_attention_forward_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                                  const __nv_bfloat16* V, __nv_bfloat16* O,
                                                  float* L, int batch_size, int num_heads,
                                                  int seq_len, int head_dim, float scale,
                                                  bool causal, cudaStream_t stream);

CUFLASH_EXPORT int cuflash_attention_backward_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                                   const __nv_bfloat16* V, const __nv_bfloat16* O,
                                                   const float* L, const __nv_bfloat16* dO,
                                                   __nv_bfloat16* dQ, __nv_bfloat16* dK,
                                                   __nv_bfloat16* dV, int batch_size, int num_heads,
                                                   int seq_len, int head_dim, float scale,
                                                   bool causal, cudaStream_t stream);

// Returns a human-readable string for the given error code (integer cast of FlashAttentionError).
CUFLASH_EXPORT const char* cuflash_error_string(int error_code);

#ifdef __cplusplus
}
#endif
