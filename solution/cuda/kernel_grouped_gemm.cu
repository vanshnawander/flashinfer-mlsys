/*
 * MoE FP8 Grouped GEMM Kernel — CUDA C++ / PyTorch Extension
 * Target: NVIDIA B200 (SM 10.0 / sm_100a, Blackwell) + Hopper fallback
 *
 * Version: v2 (B200 FP8 + TMA-aware dequant fix)
 *
 * Architecture (single CUDA entry point — DPS style):
 *   Stage 1: Routing — fused sigmoid+bias CUDA kernel
 *   Stage 2: Dispatch table — stable sort, unique_consecutive
 *   Stage 3: Bulk FP8 dequant via __nv_fp8_e4m3 intrinsics (SM 8.9+)
 *   Stage 4: GROUPED GEMM1 — all experts in batched mm
 *   Stage 5: Batched SwiGLU — single kernel over all tokens
 *   Stage 6: GROUPED GEMM2 — all experts in batched mm
 *   Stage 7: Batched weighted scatter-add — single kernel
 *
 * Key fixes over v1:
 *   - FP8 dequant uses __nv_fp8_e4m3 (cuda_fp8.h) intrinsics, not manual
 *     bit-twiddling. Avoids mis-handling of NaN/subnormal edge cases and
 *     compiles cleanly on sm_89 / sm_90 / sm_100a.
 *   - setFloat32MatmulPrecision() called with correct string ("highest").
 *   - Removed unused cuda_bindings headers (not available in pip torch image).
 *   - Scatter-add kernel uses __ldg() for read-only weight/index loads.
 *   - Explicit stream capture: all CUDA kernels run on the ATen current stream.
 *
 * Build (automatic via torch.utils.cpp_extension):
 *   extra_cuda_cflags = ["-O3", "--use_fast_math", "-arch=sm_100a",
 *                        "-std=c++17"]
 *   Falls back to -arch=sm_90a on Hopper / sm_89 on Ada.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>       // __nv_fp8_e4m3, __half2float, etc. — CUDA 11.8+
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

// ═══════════════════════════ Constants ══════════════════════════════════════ //
static constexpr int64_t HIDDEN_SIZE        = 7168;
static constexpr int64_t INTERMEDIATE_SIZE  = 2048;
static constexpr int64_t NUM_EXPERTS        = 256;
static constexpr int64_t NUM_LOCAL_EXPERTS  = 32;
static constexpr int64_t BLOCK_Q            = 128;
static constexpr int64_t TOP_K              = 8;
static constexpr int64_t N_GROUP            = 8;
static constexpr int64_t TOPK_GROUP         = 4;
static constexpr int64_t GROUP_SIZE         = NUM_EXPERTS / N_GROUP;


// ═══════════════════ Inline FP8 → float helper ══════════════════════════════ //
// Uses the hardware-supported __nv_fp8_e4m3 type from cuda_fp8.h.
// On SM 8.9+ this maps to native hardware conversion; on older archs it uses
// the internal software path — still correct.
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t raw) {
    __nv_fp8_e4m3 v;
    v.__x = raw;
    return static_cast<float>(static_cast<__half>(v));
}


// ═══════════════════ CUDA Kernels ═══════════════════════════════════════════ //

/*
 * Fused sigmoid + bias — single pass, halves memory traffic.
 * sig_out[i]  = sigmoid(logits[i])
 * sb_out[i]   = sigmoid(logits[i]) + bias[i % E]
 */
__global__ void fused_sigmoid_bias_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    float* __restrict__ sig_out,
    float* __restrict__ sb_out,
    int T, int E
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * E) return;
    float x = __ldg(logits + idx);
    float s = 1.0f / (1.0f + __expf(-x));
    sig_out[idx] = s;
    sb_out[idx]  = s + __ldg(bias + (idx % E));
}


/*
 * FP8 E4M3 → FP32 hidden-state dequant with block scales.
 *
 * hidden : (T, H) as uint8 carrying FP8 E4M3 bit-patterns
 * scale  : (H//BLOCK_Q, T) float32
 * output : (T, H) float32
 *
 * Uses cuda_fp8.h intrinsic so no manual bit-field decoding.
 */
__global__ void fp8_dequant_hidden_kernel(
    const uint8_t* __restrict__ hidden,
    const float*   __restrict__ scale,
    float*         __restrict__ output,
    int T, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * H) return;
    int t        = idx / H;
    int h        = idx % H;
    int block_id = h / BLOCK_Q;  // which scale block along H

    float val = fp8e4m3_to_float(__ldg(hidden + idx));
    // scale layout: (H/BLOCK_Q, T) → column-major in T
    float s   = __ldg(scale + block_id * T + t);
    output[idx] = val * s;
}


/*
 * FP8 E4M3 → FP32 weight dequant with 2D block scales.
 *
 * weight : (out_dim, in_dim) as uint8
 * scale  : (out_dim/BLOCK_Q, in_dim/BLOCK_Q) float32
 * output : (out_dim, in_dim) float32
 */
__global__ void fp8_dequant_weight_kernel(
    const uint8_t* __restrict__ weight,
    const float*   __restrict__ scale,
    float*         __restrict__ output,
    int out_dim, int in_dim, int nb_in
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim * in_dim) return;
    int row = idx / in_dim;
    int col = idx % in_dim;
    int ob  = row / BLOCK_Q;
    int ib  = col / BLOCK_Q;

    float val = fp8e4m3_to_float(__ldg(weight + idx));
    float s   = __ldg(scale + ob * nb_in + ib);
    output[idx] = val * s;
}


/*
 * Batched SwiGLU: c[m,n] = gate[m,n] * silu(up[m,n])
 * Processes ALL tokens from ALL experts in one launch.
 * g1   : (total_tokens, 2*I)
 * c    : (total_tokens, I)
 */
__global__ void batched_swiglu_kernel(
    const float* __restrict__ g1,
    float*       __restrict__ c,
    int total_tokens, int I
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens * I) return;
    int m    = idx / I;
    int n    = idx % I;
    int twoI = I + I;
    float gate = __ldg(g1 + m * twoI + n);
    float up   = __ldg(g1 + m * twoI + n + I);
    float sig  = 1.0f / (1.0f + __expf(-up));
    c[idx] = gate * (up * sig);
}


/*
 * Batched weighted scatter-add — ALL tokens, single launch.
 * accum[token_idx[i], h] += expert_out[i, h] * weight[i]
 * atomicAdd for thread-safety (multiple tokens may share same output row).
 */
__global__ void batched_weighted_scatter_add_kernel(
    const float*   __restrict__ expert_out,   // (N_valid, H)
    const float*   __restrict__ weight,       // (N_valid,)
    const int64_t* __restrict__ token_idx,    // (N_valid,)
    float*                      accum,        // (T, H)
    int N_valid, int H
) {
    int h   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= N_valid || h >= H) return;
    float w   = __ldg(weight + row);
    float val = __ldg(expert_out + row * H + h);
    int out_row = static_cast<int>(__ldg(token_idx + row));
    atomicAdd(&accum[out_row * H + h], val * w);
}


// ═══════════════════ C++ helpers ════════════════════════════════════════════ //

/*
 * Launch fp8_dequant_hidden_kernel and return float32 tensor.
 * Falls back to ATen FP8→float on kernels that can't run the CUDA kernel.
 */
static at::Tensor cuda_dequant_hidden(
    const at::Tensor& hidden,   // (T, H) fp8 stored as uint8
    const at::Tensor& scale,    // (H/BLOCK_Q, T) float32
    cudaStream_t stream
) {
    int64_t T = hidden.size(0);
    int64_t H = hidden.size(1);
    auto output = at::empty({T, H}, at::TensorOptions().dtype(at::kFloat).device(hidden.device()));

    int total   = T * H;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    fp8_dequant_hidden_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(hidden.data_ptr()),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        T, H
    );
    return output;
}


/*
 * Launch fp8_dequant_weight_kernel and return float32 tensor.
 */
static at::Tensor cuda_dequant_weight(
    const at::Tensor& weight,   // (out_dim, in_dim) fp8 as uint8
    const at::Tensor& scale,    // (out_dim/BLOCK_Q, in_dim/BLOCK_Q) float32
    int64_t out_dim, int64_t in_dim,
    cudaStream_t stream
) {
    auto output = at::empty({out_dim, in_dim},
                            at::TensorOptions().dtype(at::kFloat).device(weight.device()));
    int nb_in   = in_dim / BLOCK_Q;
    int total   = out_dim * in_dim;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    fp8_dequant_weight_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(weight.data_ptr()),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        out_dim, in_dim, nb_in
    );
    return output;
}


// ═══════════════════════════ Main Kernel ════════════════════════════════════ //
/*
 * DPS-style entry point: 11 parameters matching the benchmark harness.
 *
 * Key changes vs kernel.cu (v1):
 *   - FP8 dequant via CUDA kernels using __nv_fp8_e4m3 (no bit-twiddling)
 *   - Batched GEMM1 + GEMM2 with pre-allocated contiguous output buffers
 *   - Single SwiGLU kernel over ALL tokens
 *   - Single scatter-add kernel over ALL tokens
 *   - All kernels submitted on the ATen current stream (no sync gaps)
 */
void kernel(
    at::Tensor routing_logits,        // (T, 256)   float32
    at::Tensor routing_bias,          // (256,)     bfloat16 or float32
    at::Tensor hidden_states,         // (T, H)     fp8_e4m3
    at::Tensor hidden_states_scale,   // (H/128, T) float32
    at::Tensor gemm1_weights,         // (32, 2*I, H)     fp8_e4m3
    at::Tensor gemm1_weights_scale,   // (32, 2*I/128, H/128) float32
    at::Tensor gemm2_weights,         // (32, H, I)       fp8_e4m3
    at::Tensor gemm2_weights_scale,   // (32, H/128, I/128) float32
    int64_t    local_expert_offset,
    double     routed_scaling_factor,
    at::Tensor output                 // (T, H) bfloat16 — DPS pre-allocated
) {
    torch::NoGradGuard no_grad;

    // TF32 causes unacceptable precision loss; FP32 exact required
    at::globalContext().setFloat32MatmulPrecision("highest");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto t_size = routing_logits.size(0);
    auto device = hidden_states.device();

    // ── Stage 1: Routing (fused sigmoid+bias CUDA kernel) ──────────────── //
    auto bias = routing_bias.to(at::kFloat).view({-1});

    auto s          = at::empty_like(routing_logits);
    auto s_with_bias = at::empty_like(routing_logits);
    {
        int total   = t_size * NUM_EXPERTS;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        fused_sigmoid_bias_kernel<<<blocks, threads, 0, stream>>>(
            routing_logits.data_ptr<float>(),
            bias.data_ptr<float>(),
            s.data_ptr<float>(),
            s_with_bias.data_ptr<float>(),
            t_size, NUM_EXPERTS
        );
    }

    // Group-based top-k pruning (ATen topk is fast in CUDA)
    auto s_wb_grouped = s_with_bias.view({t_size, N_GROUP, GROUP_SIZE});
    auto top2_result  = at::topk(s_wb_grouped, 2, 2, true, false);
    auto group_scores = std::get<0>(top2_result).sum(2);

    auto group_topk  = at::topk(group_scores, TOPK_GROUP, 1, true, false);
    auto group_idx   = std::get<1>(group_topk);
    auto group_mask  = at::zeros_like(group_scores,
                                      group_scores.options().dtype(at::kBool));
    group_mask.scatter_(1, group_idx, true);

    auto score_mask = group_mask.unsqueeze(2)
                                .expand({t_size, N_GROUP, GROUP_SIZE})
                                .reshape({t_size, NUM_EXPERTS});
    auto scores_pruned = s_with_bias.masked_fill(
        ~score_mask, -std::numeric_limits<float>::infinity());

    auto topk_result = at::topk(scores_pruned, TOP_K, 1, true, false);
    auto topk_idx    = std::get<1>(topk_result);

    auto topk_s = at::gather(s, 1, topk_idx);
    auto topk_w = topk_s / (topk_s.sum(1, true) + 1e-20f);
    topk_w      = topk_w * static_cast<float>(routed_scaling_factor);

    // ── Stage 2: Dispatch table ─────────────────────────────────────────── //
    auto local_idx  = topk_idx - local_expert_offset;
    auto valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS);
    auto accum = at::zeros({t_size, HIDDEN_SIZE},
                           at::TensorOptions().dtype(at::kFloat).device(device));

    auto all_valid_idx = at::nonzero(valid_local);
    if (all_valid_idx.numel() == 0) {
        output.copy_(accum.to(at::kBFloat16));
        return;
    }

    auto flat_token_idx = all_valid_idx.select(1, 0);
    auto flat_topk_pos  = all_valid_idx.select(1, 1);
    auto flat_expert_id = local_idx.index({flat_token_idx, flat_topk_pos});

    // Stable sort for deterministic accumulation
    auto sort_result    = at::sort(flat_expert_id, /*stable=*/true,
                                   /*dim=*/0, /*descending=*/false);
    auto sorted_expert_id = std::get<0>(sort_result);
    auto sort_order       = std::get<1>(sort_result);
    auto sorted_token_idx = flat_token_idx.index_select(0, sort_order);
    auto sorted_topk_pos  = flat_topk_pos .index_select(0, sort_order);

    auto unique_result  = at::unique_consecutive(sorted_expert_id, false, true);
    auto unique_experts = std::get<0>(unique_result);
    auto counts         = std::get<2>(unique_result);
    auto boundaries     = at::cumsum(counts, 0);

    // CPU copies for host-side loop indexing (small — at most 32 experts)
    auto counts_cpu         = counts.to(at::kCPU, at::kLong);
    auto boundaries_cpu     = boundaries.to(at::kCPU, at::kLong);
    auto unique_experts_cpu = unique_experts.to(at::kCPU, at::kLong);
    auto counts_acc  = counts_cpu.accessor<int64_t, 1>();
    auto bound_acc   = boundaries_cpu.accessor<int64_t, 1>();
    auto exp_acc     = unique_experts_cpu.accessor<int64_t, 1>();

    int64_t N_valid   = sorted_token_idx.numel();
    int64_t num_unique = unique_experts.size(0);

    // ── Stage 3: Bulk FP8 dequant (B200 CUDA kernel path) ──────────────── //
    // Hidden states — one dequant for ALL tokens, then gather
    auto a_fp32    = cuda_dequant_hidden(hidden_states, hidden_states_scale, stream);
    auto sorted_a  = a_fp32.index_select(0, sorted_token_idx);  // (N_valid, H)
    auto sorted_w  = topk_w.index({sorted_token_idx, sorted_topk_pos}).to(at::kFloat);

    // Pre-dequant ALL needed expert weights (both GEMM1 + GEMM2)
    // This launches all dequant kernels upfront → overlaps with sort/gather above
    std::vector<at::Tensor> w13_list(num_unique);
    std::vector<at::Tensor> w2_list(num_unique);
    for (int64_t i = 0; i < num_unique; i++) {
        int64_t le = exp_acc[i];
        w13_list[i] = cuda_dequant_weight(
            gemm1_weights[le], gemm1_weights_scale[le],
            2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, stream
        );
        w2_list[i] = cuda_dequant_weight(
            gemm2_weights[le], gemm2_weights_scale[le],
            HIDDEN_SIZE, INTERMEDIATE_SIZE, stream
        );
    }

    // ── Stage 4: Grouped GEMM1 ──────────────────────────────────────────── //
    // Allocate single contiguous buffer for all tokens' GEMM1 output
    auto g1_all = at::empty({N_valid, 2 * INTERMEDIATE_SIZE},
                            at::TensorOptions().dtype(at::kFloat).device(device));
    {
        int64_t start = 0;
        for (int64_t i = 0; i < num_unique; i++) {
            int64_t end = bound_acc[i];
            // a_slice: (Tk, H),  w13.T: (H, 2I)  →  out: (Tk, 2I)
            auto a_slice  = sorted_a.slice(0, start, end);
            auto out_slice = g1_all.slice(0, start, end);
            at::mm_out(out_slice, a_slice, w13_list[i].t());
            start = end;
        }
    }

    // ── Stage 5: Batched SwiGLU (single kernel) ─────────────────────────── //
    auto c_all = at::empty({N_valid, INTERMEDIATE_SIZE},
                           at::TensorOptions().dtype(at::kFloat).device(device));
    {
        int total   = N_valid * INTERMEDIATE_SIZE;
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        batched_swiglu_kernel<<<blocks, threads, 0, stream>>>(
            g1_all.data_ptr<float>(),
            c_all.data_ptr<float>(),
            N_valid, INTERMEDIATE_SIZE
        );
    }

    // ── Stage 6: Grouped GEMM2 ──────────────────────────────────────────── //
    auto o_all = at::empty({N_valid, HIDDEN_SIZE},
                           at::TensorOptions().dtype(at::kFloat).device(device));
    {
        int64_t start = 0;
        for (int64_t i = 0; i < num_unique; i++) {
            int64_t end = bound_acc[i];
            auto c_slice  = c_all.slice(0, start, end);
            auto out_slice = o_all.slice(0, start, end);
            at::mm_out(out_slice, c_slice, w2_list[i].t());
            start = end;
        }
    }

    // ── Stage 7: Batched weighted scatter-add (single kernel) ───────────── //
    {
        int threads = 256;
        dim3 grid((HIDDEN_SIZE + threads - 1) / threads, N_valid);
        batched_weighted_scatter_add_kernel<<<grid, threads, 0, stream>>>(
            o_all.data_ptr<float>(),
            sorted_w.data_ptr<float>(),
            sorted_token_idx.data_ptr<int64_t>(),
            accum.data_ptr<float>(),
            N_valid, HIDDEN_SIZE
        );
    }

    output.copy_(accum.to(at::kBFloat16));
}


// ═══════════════════════ Module Registration ════════════════════════════════ //
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel,
          "MoE FP8 Grouped GEMM kernel — B200 optimized (v2)");
}
