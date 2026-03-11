/*
 * CUDA kernels for fused MoE (DeepSeek-V3 style)
 * moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
 *
 * Provides two device kernels:
 *   1. dequant_hidden_fp8  — FP8 block-scale dequantization of hidden states
 *   2. swiglu_activation   — SwiGLU = silu(gate) * up
 *
 * Routing, GEMMs, and expert dispatch are handled in Python (binding.py)
 * for correctness-first approach. These kernels are the hot inner loops.
 *
 * Target: NVIDIA B200 (Blackwell), compute capability 10.0+
 * Memory: 192 GB HBM3e @ 8 TB/s, 228 KB shared memory per SM
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cmath>

// ═══════════════════════════ Constants ═══════════════════════════════════════ //
constexpr int BLOCK_Q = 128;     // quantization block size

// ═══════════════════════════ Kernel 1: FP8 Dequant Hidden ════════════════════ //
/*
 * Dequantize FP8 hidden states using block scales.
 *
 * Layout:
 *   x:     [T, H]        float8_e4m3fn  (hidden states)
 *   scale: [H/128, T]    float32        (block scales, TRANSPOSED)
 *   out:   [T, H]        float32        (dequantized output)
 *
 * Each thread handles one element:
 *   out[t, h] = (float)x[t, h] * scale[h / 128, t]
 *
 * Grid: (cdiv(H, blockDim.x), cdiv(T, blockDim.y))
 */
extern "C" __global__
void dequant_hidden_fp8(
    const __nv_fp8_e4m3* __restrict__ x,      // [T, H]
    const float*         __restrict__ scale,   // [H/128, T]
    float*               __restrict__ out,     // [T, H]
    int T,
    int H,
    int num_h_blocks  // H / 128
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    if (t >= T || h >= H) return;

    // Read FP8 value and cast to float
    int idx = t * H + h;
    float val = static_cast<float>(x[idx]);

    // Scale layout: [H/128, T] row-major → scale[h_block * T + t]
    int h_block = h / BLOCK_Q;
    float s = scale[h_block * T + t];

    out[idx] = val * s;
}

// ═══════════════════════════ Kernel 2: SwiGLU ════════════════════════════════ //
/*
 * SwiGLU activation on concatenated GEMM1 output.
 *
 * Input:  g1 [rows, 2*I]  float32   — first I cols are "up", next I cols are "gate"
 * Output: c  [rows, I]    float32
 *
 * c[m, n] = g1[m, n] * silu(g1[m, n + I])
 * where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Grid: (cdiv(I, blockDim.x), cdiv(rows, blockDim.y))
 */
extern "C" __global__
void swiglu_activation(
    const float* __restrict__ g1,   // [rows, 2*I]
    float*       __restrict__ c,    // [rows, I]
    int rows,
    int I   // intermediate_size
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= rows || n >= I) return;

    int two_I = 2 * I;

    // X1 (up) = g1[m, n], X2 (gate) = g1[m, n + I]
    float x1 = g1[m * two_I + n];
    float x2 = g1[m * two_I + n + I];

    // silu(x2) = x2 * sigmoid(x2)
    float sigmoid_x2 = 1.0f / (1.0f + expf(-x2));
    float silu_x2 = x2 * sigmoid_x2;

    c[m * I + n] = x1 * silu_x2;
}

// ═══════════════════════════ Kernel 3: Weight Dequant Block ═════════════════ //
/*
 * Dequantize weight tensor using block scales.
 *
 * w:     [R, C]          float8_e4m3fn
 * scale: [R/128, C/128]  float32
 * out:   [R, C]          float32
 *
 * out[r, c] = (float)w[r, c] * scale[r/128, c/128]
 *
 * Grid: (cdiv(C, blockDim.x), cdiv(R, blockDim.y))
 */
extern "C" __global__
void dequant_weight_fp8(
    const __nv_fp8_e4m3* __restrict__ w,      // [R, C]
    const float*         __restrict__ scale,   // [R/128, C/128]
    float*               __restrict__ out,     // [R, C]
    int R,
    int C,
    int num_c_blocks   // C / 128
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= R || c >= C) return;

    int idx = r * C + c;
    float val = static_cast<float>(w[idx]);

    int r_block = r / BLOCK_Q;
    int c_block = c / BLOCK_Q;
    float s = scale[r_block * num_c_blocks + c_block];

    out[idx] = val * s;
}

// ═══════════════════════════ Kernel 4: BF16 Cast ════════════════════════════ //
/*
 * Cast float32 accumulator to bfloat16 output.
 *
 * Grid: (cdiv(N, blockDim.x), cdiv(M, blockDim.y))
 */
extern "C" __global__
void cast_fp32_to_bf16(
    const float*    __restrict__ input,    // [M, N]
    __nv_bfloat16*  __restrict__ output,   // [M, N]
    int M,
    int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) return;

    int idx = m * N + n;
    output[idx] = __float2bfloat16(input[idx]);
}
