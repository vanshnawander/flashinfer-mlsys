"""
Python binding for the fused MoE CUDA kernel.
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Uses PyTorch for routing + GEMMs (correctness-first) and can later
integrate compiled CUDA kernels via torch.utils.cpp_extension or TVM FFI
for the dequant/SwiGLU hot paths.

DPS (Destination Passing Style): The benchmark framework passes
a pre-allocated output tensor as the last argument.
"""

import torch
import math


# ───────────────────────────── geometry constants ──────────────────────────── #
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
BLOCK_Q = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = NUM_EXPERTS // N_GROUP   # 32


# ─────────────────── FP8 dequant helpers (pure PyTorch fallback) ──────────── #
def _dequant_hidden_states(hidden_states, hidden_states_scale):
    """
    Dequantize FP8 hidden states with transposed block scales.
    hidden_states: [T, H] fp8
    hidden_states_scale: [H/128, T] fp32
    """
    T, H = hidden_states.shape
    a_fp32 = hidden_states.to(torch.float32)                           # [T, H]
    s = hidden_states_scale.to(torch.float32)                          # [H/128, T]
    s_th = s.permute(1, 0).contiguous()                                # [T, H/128]
    s_expanded = (s_th.unsqueeze(-1)
                  .expand(T, H // BLOCK_Q, BLOCK_Q)
                  .reshape(T, H)
                  .contiguous())                                        # [T, H]
    return a_fp32 * s_expanded


def _dequant_w13_local(w13_e, s13_e):
    """
    Dequant GEMM1 weights for one expert.
    w13_e: [2I, H] fp8,  s13_e: [(2I)/128, H/128] fp32
    """
    n_out = 2 * INTERMEDIATE_SIZE // BLOCK_Q
    n_h = HIDDEN_SIZE // BLOCK_Q
    w = w13_e.to(torch.float32).view(n_out, BLOCK_Q, n_h, BLOCK_Q)
    s = s13_e.to(torch.float32).view(n_out, 1, n_h, 1)
    return (w * s).reshape(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)


def _dequant_w2_local(w2_e, s2_e):
    """
    Dequant GEMM2 weights for one expert.
    w2_e: [H, I] fp8,  s2_e: [H/128, I/128] fp32
    """
    n_h = HIDDEN_SIZE // BLOCK_Q
    n_i = INTERMEDIATE_SIZE // BLOCK_Q
    w = w2_e.to(torch.float32).view(n_h, BLOCK_Q, n_i, BLOCK_Q)
    s = s2_e.to(torch.float32).view(n_h, 1, n_i, 1)
    return (w * s).reshape(HIDDEN_SIZE, INTERMEDIATE_SIZE)


def _swiglu(g1):
    """
    SwiGLU activation.
    g1: [Tk, 2*I] — first I cols = up, next I cols = gate
    Returns: [Tk, I]
    """
    x1 = g1[:, :INTERMEDIATE_SIZE]           # up
    x2 = g1[:, INTERMEDIATE_SIZE:]           # gate
    return x1 * torch.nn.functional.silu(x2)


# ═══════════════════════════════ MAIN KERNEL ══════════════════════════════════ #
@torch.no_grad()
def kernel(
    # ── inputs (order must match Definition.inputs) ──
    routing_logits: torch.Tensor,       # [T, 256] fp32
    routing_bias: torch.Tensor,         # [256]    bf16
    hidden_states: torch.Tensor,        # [T, H]   fp8
    hidden_states_scale: torch.Tensor,  # [H/128, T] fp32
    gemm1_weights: torch.Tensor,        # [E_local, 2I, H] fp8
    gemm1_weights_scale: torch.Tensor,  # [E_local, (2I)/128, H/128] fp32
    gemm2_weights: torch.Tensor,        # [E_local, H, I] fp8
    gemm2_weights_scale: torch.Tensor,  # [E_local, H/128, I/128] fp32
    local_expert_offset: int,
    routed_scaling_factor: float,
    # ── DPS output (pre-allocated by framework) ──
    output: torch.Tensor,               # [T, H] bf16
):
    """
    Fused MoE layer (DeepSeek-V3 style) — CUDA/PyTorch DPS kernel.
    Correctness-first: uses PyTorch ops for routing + GEMMs.
    CUDA kernels (kernel.cu) can be integrated for dequant/SwiGLU later.
    """
    t_size = routing_logits.shape[0]
    local_start = int(local_expert_offset)
    device = hidden_states.device

    # ── Stage 1: FP8 dequant ──
    a = _dequant_hidden_states(hidden_states, hidden_states_scale)

    # ── Stage 2: DeepSeek-V3 no-aux routing ──
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    s_wb_grouped = s_with_bias.view(t_size, N_GROUP, GROUP_SIZE)
    top2_vals = torch.topk(s_wb_grouped, k=2, dim=2,
                           largest=True, sorted=False).values
    group_scores = top2_vals.sum(dim=2)

    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1,
                           largest=True, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)

    score_mask = (group_mask
                  .unsqueeze(2)
                  .expand(t_size, N_GROUP, GROUP_SIZE)
                  .reshape(t_size, NUM_EXPERTS))
    scores_pruned = s_with_bias.masked_fill(~score_mask, float("-inf"))

    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1,
                          largest=True, sorted=False).indices

    topk_s = torch.gather(s, 1, topk_idx)
    topk_w = topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * float(routed_scaling_factor)

    # ── Stage 3: Local expert compute ──
    accum = torch.zeros((t_size, HIDDEN_SIZE), dtype=torch.float32, device=device)

    local_idx = topk_idx - local_start
    valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)

    for le in range(NUM_LOCAL_EXPERTS):
        sel = valid_local & (local_idx == le)
        if not torch.any(sel):
            continue

        token_idx, topk_pos = torch.nonzero(sel, as_tuple=True)
        if token_idx.numel() == 0:
            continue

        a_e = a.index_select(0, token_idx)

        w13_e = _dequant_w13_local(gemm1_weights[le],
                                   gemm1_weights_scale[le])
        w2_e  = _dequant_w2_local(gemm2_weights[le],
                                  gemm2_weights_scale[le])

        g1 = torch.matmul(a_e, w13_e.t())
        c = _swiglu(g1)
        o = torch.matmul(c, w2_e.t())

        w_tok = topk_w[token_idx, topk_pos].to(torch.float32)
        accum.index_add_(0, token_idx, o * w_tok.unsqueeze(1))

    # ── Write into DPS output ──
    output.copy_(accum.to(torch.bfloat16))
