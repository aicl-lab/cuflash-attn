// Forward Pass Tests
// Feature: cuflash-attn, Property 1: 前向传播数值等价性

#include <gtest/gtest.h>
#if CUFLASH_ENABLE_RAPIDCHECK
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#endif
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuflash/flash_attention.h"

namespace cuflash {
namespace test {

// CPU reference implementation of standard attention
void reference_attention_forward(const std::vector<float>& Q, const std::vector<float>& K,
                                 const std::vector<float>& V, std::vector<float>& O, int batch_size,
                                 int num_heads, int seq_len, int head_dim, float scale,
                                 bool causal) {
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            int bh_offset = (b * num_heads + h) * seq_len * head_dim;

            for (int i = 0; i < seq_len; i++) {
                // Compute attention scores for row i
                std::vector<float> scores(seq_len);
                float max_score = -INFINITY;

                for (int j = 0; j < seq_len; j++) {
                    if (causal && j > i) {
                        scores[j] = -INFINITY;
                    } else {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            dot +=
                                Q[bh_offset + i * head_dim + d] * K[bh_offset + j * head_dim + d];
                        }
                        scores[j] = dot * scale;
                    }
                    max_score = std::max(max_score, scores[j]);
                }

                // Softmax
                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < seq_len; j++) {
                    scores[j] /= sum_exp;
                }

                // Weighted sum of V
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        sum += scores[j] * V[bh_offset + j * head_dim + d];
                    }
                    O[bh_offset + i * head_dim + d] = sum;
                }
            }
        }
    }
}

// Helper to compute max absolute difference
static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

// CPU reference for the logsumexp output L[b, h, i] = m_i + log(sum_j exp(s_ij - m_i)),
// where s_ij = scale * <Q_i, K_j> and m_i = max_j s_ij. This is the quantity the
// forward kernel stores for use by the backward pass.
void reference_logsumexp(const std::vector<float>& Q, const std::vector<float>& K,
                         std::vector<float>& L, int batch_size, int num_heads, int seq_len,
                         int head_dim, float scale, bool causal) {
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            int bh_offset = (b * num_heads + h) * seq_len * head_dim;
            int l_offset = (b * num_heads + h) * seq_len;

            for (int i = 0; i < seq_len; i++) {
                float max_score = -INFINITY;
                for (int j = 0; j < seq_len; j++) {
                    if (causal && j > i)
                        continue;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += Q[bh_offset + i * head_dim + d] * K[bh_offset + j * head_dim + d];
                    }
                    max_score = std::max(max_score, dot * scale);
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    if (causal && j > i)
                        continue;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += Q[bh_offset + i * head_dim + d] * K[bh_offset + j * head_dim + d];
                    }
                    sum_exp += std::exp(dot * scale - max_score);
                }

                L[l_offset + i] = max_score + std::log(sum_exp);
            }
        }
    }
}

// Basic forward test
TEST(ForwardTest, BasicSmall) {
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 8;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    // Initialize with random values
    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    std::vector<float> h_O(qkv_size), h_L(l_size);
    std::vector<float> ref_O(qkv_size);

    // Reference computation
    reference_attention_forward(h_Q, h_K, h_V, ref_O, batch_size, num_heads, seq_len, head_dim,
                                scale, false);

    // GPU computation
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    cudaDeviceSynchronize();
    cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float diff = max_abs_diff(h_O, ref_O);
    EXPECT_LT(diff, 1e-3f) << "Max difference: " << diff;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

TEST(ForwardTest, CausalMultiHeadHeadDim128) {
    const int batch_size = 2;
    const int num_heads = 2;
    const int seq_len = 16;
    const int head_dim = 128;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    std::vector<float> h_O(qkv_size), h_L(l_size), ref_O(qkv_size);
    reference_attention_forward(h_Q, h_K, h_V, ref_O, batch_size, num_heads, seq_len, head_dim,
                                scale, true);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, true, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    cudaDeviceSynchronize();
    cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    float diff = max_abs_diff(h_O, ref_O);
    EXPECT_LT(diff, 2e-3f) << "Max difference: " << diff;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

// The forward pass must store a correct logsumexp L, not just a correct O:
// the backward pass reconstructs softmax probabilities as exp(S - L), so a
// wrong L silently corrupts every gradient. Verify L directly against a
// reference instead of relying on the backward test to catch it indirectly.
TEST(ForwardTest, LogSumExpMatchesReference) {
    const int batch_size = 2;
    const int num_heads = 2;
    const int seq_len = 16;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    std::vector<float> h_L(l_size), ref_L(l_size);
    reference_logsumexp(h_Q, h_K, ref_L, batch_size, num_heads, seq_len, head_dim, scale, false);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    cudaDeviceSynchronize();
    cudaMemcpy(h_L.data(), d_L, l_size * sizeof(float), cudaMemcpyDeviceToHost);

    float diff = max_abs_diff(h_L, ref_L);
    EXPECT_LT(diff, 1e-3f) << "Max logsumexp difference: " << diff;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

// Same check with causal masking enabled, where masked positions must not
// contribute to the logsumexp.
TEST(ForwardTest, LogSumExpCausalMatchesReference) {
    const int batch_size = 1;
    const int num_heads = 2;
    const int seq_len = 16;
    const int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    std::mt19937 gen(9);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    std::vector<float> h_L(l_size), ref_L(l_size);
    reference_logsumexp(h_Q, h_K, ref_L, batch_size, num_heads, seq_len, head_dim, scale, true);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, true, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    cudaDeviceSynchronize();
    cudaMemcpy(h_L.data(), d_L, l_size * sizeof(float), cudaMemcpyDeviceToHost);

    float diff = max_abs_diff(h_L, ref_L);
    EXPECT_LT(diff, 1e-3f) << "Max causal logsumexp difference: " << diff;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

// Regression: grid.y = batch_size * num_heads must be flattened to the x
// dimension. CUDA caps gridDim.y at 65535; B*H = 512*128 = 65536 > 65535 would
// make the old launch illegal (P1 correctness, historical audit T4). This test
// runs the FP32 scalar path with seq_len=1 (so num_q_blocks=1 and the whole
// grid fits in x) and checks the output against the CPU reference.
// The WMMA/FP16 path is not exercised here because seq_len=1 does not satisfy
// its tile constraints.
TEST(ForwardTest, GridYOverflowSmoke) {
    const int batch_size = 512;
    const int num_heads = 128;  // B*H = 65536 > 65535
    const int seq_len = 1;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    size_t l_size = static_cast<size_t>(batch_size) * num_heads * seq_len;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = dist(gen);
        h_K[i] = dist(gen);
        h_V[i] = dist(gen);
    }

    std::vector<float> h_O(qkv_size), h_L(l_size);
    std::vector<float> ref_O(qkv_size);

    reference_attention_forward(h_Q, h_K, h_V, ref_O, batch_size, num_heads, seq_len, head_dim,
                                scale, /*causal=*/false);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    ASSERT_EQ(cudaMalloc(&d_Q, qkv_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_K, qkv_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_V, qkv_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_O, qkv_size * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_L, l_size * sizeof(float)), cudaSuccess);

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, /*causal=*/false, 0);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);

    cudaDeviceSynchronize();
    cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    float diff = max_abs_diff(h_O, ref_O);
    EXPECT_LT(diff, 1e-3f) << "Max difference: " << diff;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

#if CUFLASH_ENABLE_RAPIDCHECK
// Property test: Forward pass numerical equivalence
// Feature: cuflash-attn, Property 1: 前向传播数值等价性
// Validates: Requirements 1.1, 1.2, 1.5, 7.5, 8.1
RC_GTEST_PROP(ForwardProperty, NumericalEquivalence, ()) {
    int batch_size = *rc::gen::inRange(1, 3);
    int num_heads = *rc::gen::inRange(1, 4);
    int seq_len = *rc::gen::inRange(1, 65);
    int head_dim = *rc::gen::element(32, 64);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    size_t qkv_size = batch_size * num_heads * seq_len * head_dim;
    size_t l_size = batch_size * num_heads * seq_len;

    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    for (size_t i = 0; i < qkv_size; i++) {
        h_Q[i] = *rc::gen::inRange(-100, 100) * 0.01f;
        h_K[i] = *rc::gen::inRange(-100, 100) * 0.01f;
        h_V[i] = *rc::gen::inRange(-100, 100) * 0.01f;
    }

    std::vector<float> h_O(qkv_size), h_L(l_size), ref_O(qkv_size);

    reference_attention_forward(h_Q, h_K, h_V, ref_O, batch_size, num_heads, seq_len, head_dim,
                                scale, false);

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));
    cudaMalloc(&d_L, l_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_forward(d_Q, d_K, d_V, d_O, d_L, batch_size, num_heads, seq_len,
                                       head_dim, scale, false, 0);
    RC_ASSERT(err == FlashAttentionError::SUCCESS);

    cudaDeviceSynchronize();
    cudaMemcpy(h_O.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    float diff = max_abs_diff(h_O, ref_O);
    RC_ASSERT(diff < 1e-3f);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_L);
}

#endif

}  // namespace test
}  // namespace cuflash
