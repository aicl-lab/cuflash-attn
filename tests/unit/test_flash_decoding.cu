// FlashDecoding (Split-KV) Forward Tests
// Feature: cuflash-attn, E3 —— decode 阶段 KV 分块并行 + 跨块归约。
//
// 验证：
//   1. flash_attention_decode 输出与 CPU 参考逐元素一致；
//   2. 不同 num_chunks 结果不变（Split-KV 归约正确性）；
//   3. FP16 路径正确；
//   4. 非法参数返回错误而非崩溃。

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

#include "cuflash/flash_attention.h"

namespace cuflash {
namespace test {

// CPU reference: single-query attention over all KV rows (decode).
static void reference_attention_decode(const std::vector<float>& Q, const std::vector<float>& K,
                                       const std::vector<float>& V, std::vector<float>& O,
                                       int batch_size, int num_heads, int seq_len, int head_dim,
                                       float scale) {
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            const int bh = b * num_heads + h;
            const int bh_q = bh * head_dim;
            const int bh_kv = bh * seq_len * head_dim;

            std::vector<float> scores(seq_len);
            float m = -INFINITY;
            for (int j = 0; j < seq_len; j++) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s += Q[bh_q + d] * K[bh_kv + j * head_dim + d];
                }
                scores[j] = s * scale;
                m = std::max(m, scores[j]);
            }
            float l = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = std::exp(scores[j] - m);
                l += scores[j];
            }
            for (int d = 0; d < head_dim; d++) {
                float o = 0.0f;
                for (int j = 0; j < seq_len; j++) o += scores[j] * V[bh_kv + j * head_dim + d];
                O[bh * head_dim + d] = o / l;
            }
        }
    }
}

static std::vector<float> random_f32(size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); i++) mx = std::max(mx, std::abs(a[i] - b[i]));
    return mx;
}

class FlashDecodingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices found.";
        }
        cudaSetDevice(0);
    }
};

TEST_F(FlashDecodingTest, MatchesCpuReferenceF32) {
    const int batch_size = 2;
    const int num_heads = 3;
    const int seq_len = 1024;
    const int head_dim = 64;
    const int num_chunks = 8;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int bh = batch_size * num_heads;

    const auto Q = random_f32(bh * head_dim, 10);
    const auto K = random_f32(bh * seq_len * head_dim, 11);
    const auto V = random_f32(bh * seq_len * head_dim, 12);

    std::vector<float> ref(bh * head_dim);
    reference_attention_decode(Q, K, V, ref, batch_size, num_heads, seq_len, head_dim, scale);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr, *dL = nullptr;
    cudaMalloc(&dQ, Q.size() * sizeof(float));
    cudaMalloc(&dK, K.size() * sizeof(float));
    cudaMalloc(&dV, V.size() * sizeof(float));
    cudaMalloc(&dO, ref.size() * sizeof(float));
    cudaMalloc(&dL, bh * sizeof(float));
    cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

    auto err = flash_attention_decode(dQ, dK, dV, dO, dL, batch_size, num_heads, seq_len, head_dim,
                                      scale, num_chunks);
    ASSERT_EQ(err, FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> out(ref.size());
    cudaMemcpy(out.data(), dO, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_LT(max_abs_diff(out, ref), 1e-3f) << "FP32 decode vs CPU reference";

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    cudaFree(dL);
}

TEST_F(FlashDecodingTest, ChunkCountInvariant) {
    const int batch_size = 1;
    const int num_heads = 4;
    const int seq_len = 2048;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int bh = batch_size * num_heads;

    const auto Q = random_f32(bh * head_dim, 20);
    const auto K = random_f32(bh * seq_len * head_dim, 21);
    const auto V = random_f32(bh * seq_len * head_dim, 22);

    std::vector<float> ref(bh * head_dim);
    reference_attention_decode(Q, K, V, ref, batch_size, num_heads, seq_len, head_dim, scale);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO1 = nullptr, *dO16 = nullptr, *dL = nullptr;
    cudaMalloc(&dQ, Q.size() * sizeof(float));
    cudaMalloc(&dK, K.size() * sizeof(float));
    cudaMalloc(&dV, V.size() * sizeof(float));
    cudaMalloc(&dO1, bh * head_dim * sizeof(float));
    cudaMalloc(&dO16, bh * head_dim * sizeof(float));
    cudaMalloc(&dL, bh * sizeof(float));
    cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_EQ(flash_attention_decode(dQ, dK, dV, dO1, dL, batch_size, num_heads, seq_len, head_dim,
                                     scale, 1),
              FlashAttentionError::SUCCESS);
    ASSERT_EQ(flash_attention_decode(dQ, dK, dV, dO16, dL, batch_size, num_heads, seq_len,
                                     head_dim, scale, 16),
              FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<float> out1(bh * head_dim), out16(bh * head_dim);
    cudaMemcpy(out1.data(), dO1, out1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out16.data(), dO16, out16.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Split-KV 归约必须与 chunk=1（= 标准 decode）数值一致，且都匹配 CPU 参考。
    EXPECT_LT(max_abs_diff(out1, ref), 1e-3f) << "chunk=1 vs CPU reference";
    EXPECT_LT(max_abs_diff(out16, ref), 1e-3f) << "chunk=16 vs CPU reference";
    EXPECT_LT(max_abs_diff(out1, out16), 1e-4f) << "chunk=1 vs chunk=16";

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO1);
    cudaFree(dO16);
    cudaFree(dL);
}

TEST_F(FlashDecodingTest, MatchesCpuReferenceHalf) {
    const int batch_size = 2;
    const int num_heads = 2;
    const int seq_len = 512;
    const int head_dim = 64;
    const int num_chunks = 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int bh = batch_size * num_heads;

    const auto Qf = random_f32(bh * head_dim, 30);
    const auto Kf = random_f32(bh * seq_len * head_dim, 31);
    const auto Vf = random_f32(bh * seq_len * head_dim, 32);

    std::vector<half> Q(Qf.size()), K(Kf.size()), V(Vf.size());
    for (size_t i = 0; i < Qf.size(); i++) Q[i] = __float2half(Qf[i]);
    for (size_t i = 0; i < Kf.size(); i++) K[i] = __float2half(Kf[i]);
    for (size_t i = 0; i < Vf.size(); i++) V[i] = __float2half(Vf[i]);

    std::vector<float> ref(bh * head_dim);
    reference_attention_decode(Qf, Kf, Vf, ref, batch_size, num_heads, seq_len, head_dim, scale);

    half *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    float* dL = nullptr;
    cudaMalloc(&dQ, Q.size() * sizeof(half));
    cudaMalloc(&dK, K.size() * sizeof(half));
    cudaMalloc(&dV, V.size() * sizeof(half));
    cudaMalloc(&dO, ref.size() * sizeof(half));
    cudaMalloc(&dL, bh * sizeof(float));
    cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * sizeof(half), cudaMemcpyHostToDevice);

    ASSERT_EQ(flash_attention_decode(dQ, dK, dV, dO, dL, batch_size, num_heads, seq_len, head_dim,
                                     scale, num_chunks),
              FlashAttentionError::SUCCESS);
    cudaDeviceSynchronize();

    std::vector<half> out(ref.size());
    cudaMemcpy(out.data(), dO, out.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    float mx = 0.0f;
    for (size_t i = 0; i < out.size(); i++) mx = std::max(mx, std::abs(__half2float(out[i]) - ref[i]));
    EXPECT_LT(mx, 1e-2f) << "FP16 decode vs CPU reference";

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    cudaFree(dL);
}

TEST_F(FlashDecodingTest, InvalidArgumentsReturnError) {
    float dummy = 0.0f;
    float* null_ptr = nullptr;  // 类型化空指针（O/L 为 float*，消歧 3 个 dtype 重载）
    // 空指针 → NULL_POINTER（在任何 kernel 启动前返回）
    EXPECT_EQ(flash_attention_decode(null_ptr, null_ptr, null_ptr, null_ptr, null_ptr, 1, 1, 128,
                                     64, 1.0f, 4),
              FlashAttentionError::NULL_POINTER);
    // 不支持的 head_dim（有效指针）→ UNSUPPORTED_DTYPE
    EXPECT_EQ(flash_attention_decode(&dummy, &dummy, &dummy, &dummy, &dummy, 1, 1, 128, 17, 1.0f,
                                     4),
              FlashAttentionError::UNSUPPORTED_HEAD_DIM);
    // batch_size=0（有效指针）→ INVALID_DIMENSION
    EXPECT_EQ(flash_attention_decode(&dummy, &dummy, &dummy, &dummy, &dummy, 0, 1, 128, 64, 1.0f,
                                     4),
              FlashAttentionError::INVALID_DIMENSION);
}

}  // namespace test
}  // namespace cuflash
