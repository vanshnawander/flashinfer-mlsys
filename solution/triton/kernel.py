"""
Triton optimized MoE kernel — Submission 5
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Key optimizations based on profiling (sub-3 data on B200):

  Bottleneck 1: Sequential expert loop = 85% of time on large-T
  → Fix: Pre-dequant all accessed weights ONCE, batch matmuls
  → Fix: Use torch.matmul for ALL GEMMs (cuBLAS is faster than our Triton GEMM)
  → Insight: Our fused GEMM tl.dot on FP32 uses CUDA cores, not tensor cores.
     cuBLAS FP32 matmul actually utilizes TF32 tensor cores on B200 = 2× faster!

  Bottleneck 2: Weight dequant done separately per expert
  → Fix: Pre-dequant weights for active experts only, reuse across tokens

  Bottleneck 3: Routing overhead (~10 PyTorch ops)
  → Fix: Minimize intermediate allocations, use in-place ops where possible

  Bottleneck 4: Per-expert kernel launches (3-5 per expert × 32 = ~100 launches)
  → Fix: Keep everything in cuBLAS (one launch per matmul, highly optimized)

Strategy shift from sub-4:
  Sub-4 tried custom Triton GEMM with on-the-fly dequant, but tl.dot on FP32
  uses CUDA cores (~60 TFLOPS) while cuBLAS uses TF32 tensor cores (~120 TFLOPS).
  Better approach: pre-dequant weights (bandwidth-cheap since FP8→FP32 is 1:4)
  then let cuBLAS handle the GEMM at TF32 speed.
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════ Geometry Constants ══════════════════════════════ #
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
BLOCK_Q = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = NUM_EXPERTS // N_GROUP   # 32


# ━━━━━━━━━━━ Triton: FP8 Hidden State Dequant ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _dequant_hidden_fp8_kernel(
    x_ptr, s_ptr, o_ptr,
    t_size, h_size,
    sx0, sx1, ss0, ss1, so0, so1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < t_size
    mask_n = offs_n < h_size
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(x_ptr + offs_m[:, None] * sx0 + offs_n[None, :] * sx1,
                mask=mask, other=0.0).to(tl.float32)
    h_block = offs_n // SCALE_BLOCK
    s = tl.load(s_ptr + h_block[None, :] * ss0 + offs_m[:, None] * ss1,
                mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1,
             x * s, mask=mask)


# ━━━━━━━━━━━ Triton: SwiGLU Activation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _swiglu_kernel(
    g1_ptr, c_ptr, rows, i_size,
    sg10, sg11, sc0, sc1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < rows
    mask_n = offs_n < i_size
    mask = mask_m[:, None] & mask_n[None, :]
    x1 = tl.load(g1_ptr + offs_m[:, None] * sg10 + offs_n[None, :] * sg11,
                 mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(g1_ptr + offs_m[:, None] * sg10 + (offs_n[None, :] + i_size) * sg11,
                 mask=mask, other=0.0).to(tl.float32)
    tl.store(c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1,
             x1 * (x2 * tl.sigmoid(x2)), mask=mask)


# ━━━━━━━━━━━ Python Launchers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
def _dequant_hidden_states(hidden_states, hidden_states_scale):
    t_size, h_size = hidden_states.shape
    out = torch.empty((t_size, h_size), device=hidden_states.device, dtype=torch.float32)
    BM, BN = 32, 128
    grid = (triton.cdiv(t_size, BM), triton.cdiv(h_size, BN))
    _dequant_hidden_fp8_kernel[grid](
        hidden_states, hidden_states_scale, out,
        t_size, h_size,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, SCALE_BLOCK=BLOCK_Q,
    )
    return out


def _swiglu(g1):
    rows = g1.shape[0]
    c = torch.empty((rows, INTERMEDIATE_SIZE), device=g1.device, dtype=torch.float32)
    BM, BN = 64, 128
    grid = (triton.cdiv(rows, BM), triton.cdiv(INTERMEDIATE_SIZE, BN))
    _swiglu_kernel[grid](
        g1, c, rows, INTERMEDIATE_SIZE,
        g1.stride(0), g1.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return c


def _swiglu_torch(g1):
    """SwiGLU using PyTorch — lower launch overhead for small batches."""
    x1 = g1[:, :INTERMEDIATE_SIZE]
    x2 = g1[:, INTERMEDIATE_SIZE:]
    return (x1 * torch.nn.functional.silu(x2)).to(torch.float32)


# ━━━━━━━━━━━ Weight Dequant (block-scale FP8 → FP32) ━━━━━━━━━━━━━━━━━━━━━━ #
def _dequant_weight(w_fp8, scale, out_dim, in_dim):
    """
    Dequant one expert's weight: w_fp32[i,j] = w_fp8[i,j] × scale[i÷128, j÷128]
    Uses view + broadcast (no repeat_interleave = less memory).
    """
    nb_out = out_dim // BLOCK_Q
    nb_in = in_dim // BLOCK_Q
    w = w_fp8.to(torch.float32).view(nb_out, BLOCK_Q, nb_in, BLOCK_Q)
    s = scale.to(torch.float32).view(nb_out, 1, nb_in, 1)
    return (w * s).reshape(out_dim, in_dim)


# ═══════════════════════════════ MAIN KERNEL ══════════════════════════════════ #
# MoE FFN layer — sits AFTER attention + layernorm in the transformer block.
# DeepSeek-V3 SwiGLU per expert:
#   GEMM1: x × [W1‖W3]ᵀ → [up‖gate]   (W1=up, W3=gate, concatenated)
#   SwiGLU: up × silu(gate) → activated
#   GEMM2: activated × W2ᵀ → output    (W2=down projection)
# Math:  y = W2ᵀ · ( (W1ᵀ·x) ⊙ silu(W3ᵀ·x) )
# ══════════════════════════════════════════════════════════════════════════════ #
@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,       # [T, 256] fp32 — raw router scores
    routing_bias: torch.Tensor,         # [256]    bf16 — load-balancing bias
    hidden_states: torch.Tensor,        # [T, H=7168]   fp8 — quantized activations
    hidden_states_scale: torch.Tensor,  # [H/128=56, T] fp32 — block scales (TRANSPOSED)
    gemm1_weights: torch.Tensor,        # [32, 4096, 7168] fp8 — cat([W1,W3]) per expert
    gemm1_weights_scale: torch.Tensor,  # [32, 32, 56] fp32 — block scales for W13
    gemm2_weights: torch.Tensor,        # [32, 7168, 2048] fp8 — W2 per expert
    gemm2_weights_scale: torch.Tensor,  # [32, 56, 16] fp32 — block scales for W2
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,               # [T, H=7168] bf16 — DPS output
):
    t_size = routing_logits.shape[0]
    local_start = int(local_expert_offset)
    device = hidden_states.device

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()

    # ── Stage 1: Dequant FP8 hidden states → FP32 (Triton kernel) ──────────── #
    a = _dequant_hidden_states(hidden_states, hidden_states_scale)

    # ── Stage 2: DeepSeek-V3 routing ───────────────────────────────────────── #
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).view(-1)

    s = torch.sigmoid(logits)                                      # [T, 256]
    s_with_bias = s + bias                                         # [T, 256]

    # Group scoring: top-2 per group → sum → top-4 groups
    s_wb_grouped = s_with_bias.view(t_size, N_GROUP, GROUP_SIZE)   # [T, 8, 32]
    top2_vals = torch.topk(s_wb_grouped, k=2, dim=2,
                           largest=True, sorted=False).values
    group_scores = top2_vals.sum(dim=2)                            # [T, 8]

    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1,
                           largest=True, sorted=False).indices     # [T, 4]
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)                        # [T, 8]

    # Mask out non-selected groups, pick top-8 experts
    score_mask = group_mask.unsqueeze(2).expand(
        t_size, N_GROUP, GROUP_SIZE).reshape(t_size, NUM_EXPERTS)
    scores_pruned = s_with_bias.masked_fill(~score_mask, float("-inf"))

    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1,
                          largest=True, sorted=False).indices      # [T, 8]

    # Normalize using UNBIASED sigmoid scores
    topk_s = torch.gather(s, 1, topk_idx)                         # [T, 8]
    topk_w = topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * float(routed_scaling_factor)                 # [T, 8]

    # ── Stage 3: Build dispatch table (single scan) ────────────────────────── #
    local_idx = topk_idx - local_start                             # [T, 8]
    valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)

    # Pre-compute per-expert token lists in one pass
    expert_token_lists = [None] * NUM_LOCAL_EXPERTS
    expert_topk_lists = [None] * NUM_LOCAL_EXPERTS
    active_experts = []

    if torch.any(valid_local):
        all_valid_idx = torch.nonzero(valid_local, as_tuple=False)
        if all_valid_idx.numel() > 0:
            flat_token_idx = all_valid_idx[:, 0]
            flat_topk_pos = all_valid_idx[:, 1]
            flat_expert_id = local_idx[flat_token_idx, flat_topk_pos]

            sort_order = torch.argsort(flat_expert_id, stable=True)
            sorted_expert_id = flat_expert_id[sort_order]
            sorted_token_idx = flat_token_idx[sort_order]
            sorted_topk_pos = flat_topk_pos[sort_order]

            unique_experts, counts = torch.unique_consecutive(
                sorted_expert_id, return_counts=True
            )
            boundaries = torch.cumsum(counts, dim=0)
            start = 0
            for i in range(unique_experts.numel()):
                le = unique_experts[i].item()
                end = boundaries[i].item()
                expert_token_lists[le] = sorted_token_idx[start:end]
                expert_topk_lists[le] = sorted_topk_pos[start:end]
                active_experts.append(le)
                start = end

    # ── Stage 4: Pre-dequant weights for ACTIVE experts only ───────────────── #
    # Pre-dequant is faster than on-the-fly because:
    # 1. cuBLAS matmul uses TF32 tensor cores (~120 TFLOPS on B200)
    # 2. Our Triton tl.dot on FP32 uses CUDA cores (~60 TFLOPS)
    # 3. Weight dequant is bandwidth-cheap: 1B FP8 → 4B FP32 (simple multiply)
    # 4. We only dequant experts that actually have tokens (often < 32)
    w13_cache = {}
    w2_cache = {}
    for le in active_experts:
        w13_cache[le] = _dequant_weight(
            gemm1_weights[le], gemm1_weights_scale[le],
            2 * INTERMEDIATE_SIZE, HIDDEN_SIZE
        )
        w2_cache[le] = _dequant_weight(
            gemm2_weights[le], gemm2_weights_scale[le],
            HIDDEN_SIZE, INTERMEDIATE_SIZE
        )

    # ── Stage 5: Expert compute ────────────────────────────────────────────── #
    accum = torch.zeros((t_size, HIDDEN_SIZE), dtype=torch.float32, device=device)

    for le in active_experts:
        token_idx = expert_token_lists[le]
        topk_pos = expert_topk_lists[le]
        Tk = token_idx.numel()

        a_e = a.index_select(0, token_idx)                         # [Tk, H]

        # GEMM1: x × [W1‖W3]ᵀ → [Tk, 4096] (cuBLAS: TF32 tensor cores)
        g1 = torch.matmul(a_e, w13_cache[le].t())                  # [Tk, 2I]

        # SwiGLU: up × silu(gate) → [Tk, 2048]
        if Tk >= 16:
            c = _swiglu(g1)                                        # Triton kernel
        else:
            c = _swiglu_torch(g1)                                  # PyTorch fallback

        # GEMM2: activated × W2ᵀ → [Tk, 7168] (cuBLAS: TF32 tensor cores)
        o = torch.matmul(c, w2_cache[le].t())                      # [Tk, H]

        # Weighted scatter-add: accum[t] += weight × expert_output
        w_tok = topk_w[token_idx, topk_pos].to(torch.float32)
        accum.index_add_(0, token_idx, o * w_tok.unsqueeze(1))

    # ── Write BF16 result into DPS output ──────────────────────────────────── #
    output.copy_(accum.to(torch.bfloat16))
