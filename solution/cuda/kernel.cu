/*
 * MoE CUDA Kernel — Optimized for B200 (SM 10.0, Blackwell)
 * moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
 *
 * Optimizations over v13 ATen-only:
 *   1. Custom CUDA kernels for element-wise ops (skip ATen dispatch overhead)
 *   2. Fused sigmoid+bias in single pass (halves memory traffic)
 *   3. No unnecessary .to() conversions (routing_logits already FP32)
 *   4. Custom CUDA SwiGLU (single kernel, no intermediate alloc)
 *   5. Fused weight-multiply + scatter-add kernel
 *   6. FP32-exact matmul (TF32 disabled for correctness)
 *   7. Stable sort for deterministic accumulation
 *
 * Entry: kernel.cu::kernel (DPS style, 11 parameters)
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <tuple>

// ═══════════════════════════ Constants ══════════════════════════════════════ //
static constexpr int64_t HIDDEN_SIZE = 7168;
static constexpr int64_t INTERMEDIATE_SIZE = 2048;
static constexpr int64_t NUM_EXPERTS = 256;
static constexpr int64_t NUM_LOCAL_EXPERTS = 32;
static constexpr int64_t BLOCK_Q = 128;
static constexpr int64_t TOP_K = 8;
static constexpr int64_t N_GROUP = 8;
static constexpr int64_t TOPK_GROUP = 4;
static constexpr int64_t GROUP_SIZE = NUM_EXPERTS / N_GROUP;


// ═══════════════════ Custom CUDA Kernels ═══════════════════════════════════ //

/*
 * Fused sigmoid + bias: out[i] = sigmoid(logits[i]) + bias[i % E]
 * Also stores raw sigmoid: sig_out[i] = sigmoid(logits[i])
 * Single pass over logits — halves memory traffic vs separate ops.
 */
__global__ void fused_sigmoid_bias_kernel(
    const float* __restrict__ logits,   // (T, E)
    const float* __restrict__ bias,     // (E,)
    float* __restrict__ sig_out,        // (T, E) — raw sigmoid
    float* __restrict__ sb_out,         // (T, E) — sigmoid + bias
    int T, int E
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * E) return;
    int e = idx % E;
    float x = logits[idx];
    float s = 1.0f / (1.0f + __expf(-x));  // fast sigmoid via CUDA intrinsic
    sig_out[idx] = s;
    sb_out[idx] = s + bias[e];
}


/*
 * SwiGLU: out[m,n] = gate[m,n] * silu(up[m,n])
 * where gate = g1[:, :I], up = g1[:, I:]
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * Single kernel — no intermediate sigmoid tensor.
 */
__global__ void swiglu_kernel(
    const float* __restrict__ g1,    // (M, 2*I)
    float* __restrict__ out,         // (M, I)
    int M, int I
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * I) return;
    int m = idx / I;
    int n = idx % I;
    int two_I = I + I;
    float gate = g1[m * two_I + n];
    float up = g1[m * two_I + n + I];
    float sig = 1.0f / (1.0f + __expf(-up));
    out[idx] = gate * (up * sig);
}


/*
 * Fused weighted scatter-add:
 * accum[token_idx[i], :] += expert_out[i, :] * weight[i]
 * Uses atomicAdd for thread-safety.
 */
__global__ void weighted_scatter_add_kernel(
    const float* __restrict__ expert_out,  // (Tk, H)
    const float* __restrict__ weight,      // (Tk,)
    const int64_t* __restrict__ token_idx, // (Tk,)
    float* __restrict__ accum,             // (T, H)
    int Tk, int H
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= Tk || h >= H) return;
    float w = weight[row];
    float val = expert_out[row * H + h];
    int out_row = static_cast<int>(token_idx[row]);
    atomicAdd(&accum[out_row * H + h], val * w);
}


// ═══════════════════ Helper Functions ═══════════════════════════════════ //

static at::Tensor dequant_hidden(
    const at::Tensor& hidden_states,
    const at::Tensor& hidden_states_scale
) {
    auto t_size = hidden_states.size(0);
    auto h_size = hidden_states.size(1);
    auto nb_h = h_size / BLOCK_Q;
    auto x = hidden_states.to(at::kFloat).view({t_size, nb_h, BLOCK_Q});
    auto s = hidden_states_scale.to(at::kFloat).t().unsqueeze(2);
    return (x * s).reshape({t_size, h_size});
}

static at::Tensor dequant_weight(
    const at::Tensor& w_fp8,
    const at::Tensor& scale,
    int64_t out_dim,
    int64_t in_dim
) {
    auto nb_out = out_dim / BLOCK_Q;
    auto nb_in = in_dim / BLOCK_Q;
    auto w = w_fp8.to(at::kFloat).view({nb_out, BLOCK_Q, nb_in, BLOCK_Q});
    auto s = scale.to(at::kFloat).view({nb_out, 1, nb_in, 1});
    return (w * s).reshape({out_dim, in_dim});
}


// ═══════════════════════════ Main Kernel ════════════════════════════════════ //
void kernel(
    at::Tensor routing_logits,        // (T, 256) float32 — ALREADY FP32
    at::Tensor routing_bias,          // (256,) bfloat16
    at::Tensor hidden_states,         // (T, H) fp8
    at::Tensor hidden_states_scale,   // (H//128, T) float32
    at::Tensor gemm1_weights,         // (32, 2*I, H) fp8
    at::Tensor gemm1_weights_scale,   // (32, 2*I//128, H//128) float32
    at::Tensor gemm2_weights,         // (32, H, I) fp8
    at::Tensor gemm2_weights_scale,   // (32, H//128, I//128) float32
    int64_t local_expert_offset,
    double routed_scaling_factor,
    at::Tensor output                 // (T, H) bfloat16 — DPS output
) {
    torch::NoGradGuard no_grad;

    // Force FP32 matmul — TF32 causes large errors on B200
    at::globalContext().setFloat32MatmulPrecision("highest");

    auto t_size = routing_logits.size(0);
    auto device = hidden_states.device();

    // ── Routing (fused CUDA kernels) ─────────────────────────────────── //
    // routing_logits is ALREADY float32 — no conversion needed
    auto bias = routing_bias.to(at::kFloat).view({-1});

    // Fused sigmoid + bias in single CUDA kernel
    auto s = at::empty_like(routing_logits);          // raw sigmoid
    auto s_with_bias = at::empty_like(routing_logits); // sigmoid + bias
    {
        int total = t_size * NUM_EXPERTS;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        fused_sigmoid_bias_kernel<<<blocks, threads>>>(
            routing_logits.data_ptr<float>(),
            bias.data_ptr<float>(),
            s.data_ptr<float>(),
            s_with_bias.data_ptr<float>(),
            t_size, NUM_EXPERTS
        );
    }

    // Group-based pruning (ATen — topk is already fast on GPU)
    auto s_wb_grouped = s_with_bias.view({t_size, N_GROUP, GROUP_SIZE});
    auto top2_result = at::topk(s_wb_grouped, 2, 2, true, false);
    auto group_scores = std::get<0>(top2_result).sum(2);

    auto group_topk = at::topk(group_scores, TOPK_GROUP, 1, true, false);
    auto group_idx = std::get<1>(group_topk);
    auto group_mask = at::zeros_like(group_scores, group_scores.options().dtype(at::kBool));
    group_mask.scatter_(1, group_idx, true);

    auto score_mask = group_mask.unsqueeze(2)
        .expand({t_size, N_GROUP, GROUP_SIZE})
        .reshape({t_size, NUM_EXPERTS});
    auto scores_pruned = s_with_bias.masked_fill(
        ~score_mask, -std::numeric_limits<float>::infinity()
    );

    auto topk_result = at::topk(scores_pruned, TOP_K, 1, true, false);
    auto topk_idx = std::get<1>(topk_result);

    // Routing weights — reuse s (raw sigmoid), no re-gather from logits
    auto topk_s = at::gather(s, 1, topk_idx);
    auto topk_w = topk_s / (topk_s.sum(1, true) + 1e-20);
    topk_w = topk_w * routed_scaling_factor;

    // ── Dispatch ───────────────────────────────────────────────────────── //
    auto local_idx = topk_idx - local_expert_offset;
    auto valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS);
    auto accum = at::zeros({t_size, HIDDEN_SIZE}, at::TensorOptions().dtype(at::kFloat).device(device));

    auto all_valid_idx = at::nonzero(valid_local);
    if (all_valid_idx.numel() == 0) {
        output.copy_(accum.to(at::kBFloat16));
        return;
    }

    auto flat_token_idx = all_valid_idx.select(1, 0);
    auto flat_topk_pos = all_valid_idx.select(1, 1);
    auto flat_expert_id = local_idx.index({flat_token_idx, flat_topk_pos});

    // Stable sort — preserves token order within same expert
    auto sort_result = at::sort(flat_expert_id, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    auto sorted_expert_id = std::get<0>(sort_result);
    auto sort_order = std::get<1>(sort_result);
    auto sorted_token_idx = flat_token_idx.index_select(0, sort_order);
    auto sorted_topk_pos = flat_topk_pos.index_select(0, sort_order);

    auto unique_result = at::unique_consecutive(sorted_expert_id, false, true);
    auto unique_experts = std::get<0>(unique_result);
    auto counts = std::get<2>(unique_result);
    auto boundaries = at::cumsum(counts, 0);

    // ── Dequant hidden states (once) ────────────────────────────────── //
    auto a_fp32 = dequant_hidden(hidden_states, hidden_states_scale);

    // ── Pre-gather all needed data (batch index_select) ─────────────── //
    auto sorted_a = a_fp32.index_select(0, sorted_token_idx);
    auto sorted_w = topk_w.index({sorted_token_idx, sorted_topk_pos}).to(at::kFloat);

    // ── Per-expert compute ──────────────────────────────────────────── //
    int64_t start = 0;
    auto num_unique = unique_experts.size(0);

    for (int64_t i = 0; i < num_unique; i++) {
        auto le = unique_experts[i].item<int64_t>();
        auto end = boundaries[i].item<int64_t>();
        int64_t Tk = end - start;

        // Slice pre-gathered data (no index_select per expert)
        auto a_e = sorted_a.slice(0, start, end);         // (Tk, H)
        auto w_e = sorted_w.slice(0, start, end);         // (Tk,)
        auto t_idx = sorted_token_idx.slice(0, start, end);

        // GEMM1: (Tk, H) × (2*I, H).T → (Tk, 2*I) via cuBLAS
        auto w13_e = dequant_weight(
            gemm1_weights[le], gemm1_weights_scale[le],
            2 * INTERMEDIATE_SIZE, HIDDEN_SIZE
        );
        auto g1 = at::matmul(a_e, w13_e.t());

        // SwiGLU via custom CUDA kernel (no intermediate alloc)
        auto c_result = at::empty({Tk, INTERMEDIATE_SIZE}, g1.options());
        {
            int total = Tk * INTERMEDIATE_SIZE;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            swiglu_kernel<<<blocks, threads>>>(
                g1.data_ptr<float>(),
                c_result.data_ptr<float>(),
                Tk, INTERMEDIATE_SIZE
            );
        }

        // GEMM2: (Tk, I) × (H, I).T → (Tk, H) via cuBLAS
        auto w2_e = dequant_weight(
            gemm2_weights[le], gemm2_weights_scale[le],
            HIDDEN_SIZE, INTERMEDIATE_SIZE
        );
        auto o_result = at::matmul(c_result, w2_e.t());

        // Fused weighted scatter-add via custom CUDA kernel
        {
            int threads = 256;
            dim3 grid((HIDDEN_SIZE + threads - 1) / threads, Tk);
            weighted_scatter_add_kernel<<<grid, threads>>>(
                o_result.data_ptr<float>(),
                w_e.data_ptr<float>(),
                t_idx.data_ptr<int64_t>(),
                accum.data_ptr<float>(),
                Tk, HIDDEN_SIZE
            );
        }

        start = end;
    }

    output.copy_(accum.to(at::kBFloat16));
}


// ═══════════════════════ Module Registration ════════════════════════════════ //
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &kernel, "MoE FP8 block-scale kernel — B200 optimized");
}
