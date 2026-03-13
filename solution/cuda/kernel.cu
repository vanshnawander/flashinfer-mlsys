/*
 * CUDA kernels for fused MoE (DeepSeek-V3 style)
 * moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
 *
 * Target: NVIDIA B200 (Blackwell)
 *
 * NOTE: FP8 data is handled as uint8_t (raw bytes) and converted via
 * a portable E4M3 decoder. This avoids dependency on cuda_fp8.h which
 * may not be available in all CUDA toolkit versions.
 *
 * The binding.py handles the heavy lifting (routing, GEMMs via cuBLAS).
 * These kernels provide accelerated element-wise operations:
 *   1. swiglu_fused        — SwiGLU activation (no intermediates)
 *   2. weighted_scatter_add — Fused weight × scatter-add
 *   3. cast_fp32_to_bf16   — FP32→BF16 output cast
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>


// ═══════════════════════ Kernel 1: Fused SwiGLU ════════════════════════════ //
/*
 * c[m, n] = g1[m, n] * silu(g1[m, n + I])
 * where silu(x) = x / (1 + exp(-x))
 *
 * Single pass: load both halves, compute in-register, store immediately.
 *
 * Block: (256, 1), Grid: (ceil(I/256), rows)
 */
extern "C" __global__
void swiglu_fused(
    const float* __restrict__ g1,
    float*       __restrict__ c,
    int rows, int I
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;

    if (m >= rows || n >= I) return;

    const int two_I = I + I;
    const float up   = g1[m * two_I + n];
    const float gate = g1[m * two_I + n + I];

    // Fused silu(gate) * up — minimal registers, no intermediates
    c[m * I + n] = up * (gate / (1.0f + expf(-gate)));
}


// ═══════════════════════ Kernel 2: Weighted Scatter-Add ══════════════════ //
/*
 * Fused: accum[token_idx[i], h] += weight[i] * expert_out[i, h]
 *
 * Block: (256, 1), Grid: (ceil(H/256), Tk)
 */
extern "C" __global__
void weighted_scatter_add(
    const float*    __restrict__ expert_output,
    const float*    __restrict__ weight,
    const int64_t*  __restrict__ token_idx,
    float*                       accum,
    int Tk, int H
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (row >= Tk || h >= H) return;

    const float w = weight[row];
    const float val = expert_output[row * H + h];
    const int out_row = static_cast<int>(token_idx[row]);

    atomicAdd(&accum[out_row * H + h], val * w);
}


// ═══════════════════════ Kernel 3: FP32→BF16 Cast ════════════════════════ //
/*
 * Portable FP32→BF16 conversion using bit manipulation.
 * Avoids dependency on cuda_bf16.h / __nv_bfloat16.
 * BF16 = upper 16 bits of FP32 (with rounding).
 *
 * Block: (256, 1), Grid: (ceil(total/256), 1)
 */
extern "C" __global__
void cast_fp32_to_bf16(
    const float*    __restrict__ input,
    unsigned short* __restrict__ output,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // BF16 = truncate FP32 to upper 16 bits (with round-to-nearest-even)
    unsigned int bits = __float_as_uint(input[idx]);
    // Round: add 0x7FFF + bit[16] for round-to-nearest-even
    unsigned int rounding = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding;
    output[idx] = static_cast<unsigned short>(bits >> 16);
}
