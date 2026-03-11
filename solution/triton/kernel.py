"""
Triton implementation for the MLSys'26 fused_moe track definition:
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Correctness-first implementation aligned with the official reference.
Uses Triton kernels for FP8 hidden-state dequantization and SwiGLU,
while routing/topk and GEMMs use PyTorch operators.

DPS (Destination Passing Style): The benchmark framework passes a
pre-allocated output tensor as the last argument. The kernel writes
the result into it in-place.
"""

import torch
import triton
import triton.language as tl


# ───────────────────────────── geometry constants ──────────────────────────── #
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
BLOCK_Q = 128              # quantization block size (DeepSeek-V3)
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = NUM_EXPERTS // N_GROUP   # 32


# ────────────────────── Triton kernel: FP8 dequant hidden ─────────────────── #
@triton.jit
def _dequant_hidden_fp8_kernel(
    x_ptr,       # [T, H] fp8
    s_ptr,       # [H/128, T] fp32   — NOTE transposed scale layout
    o_ptr,       # [T, H] fp32       — output
    t_size,      # number of tokens
    h_size,      # hidden size (7168)
    sx0, sx1,    # strides for x (row, col)
    ss0, ss1,    # strides for scale (row=h_block, col=token)
    so0, so1,    # strides for output (row, col)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    """
    Dequantize FP8 hidden states using block scales.
    Each element: output[t, h] = fp8_value[t, h] * scale[h // 128, t]
    """
    pid_m = tl.program_id(0)   # token tile
    pid_n = tl.program_id(1)   # hidden tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < t_size
    mask_n = offs_n < h_size
    mask = mask_m[:, None] & mask_n[None, :]

    # Load FP8 values and cast to FP32
    x_ptrs = x_ptr + offs_m[:, None] * sx0 + offs_n[None, :] * sx1
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Scale layout: [H/128, T] — index by (h_block, token)
    h_block = offs_n // SCALE_BLOCK   # which 128-element block
    s_ptrs = s_ptr + h_block[None, :] * ss0 + offs_m[:, None] * ss1
    s = tl.load(s_ptrs, mask=mask, other=0.0).to(tl.float32)

    out = x * s
    o_ptrs = o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1
    tl.store(o_ptrs, out, mask=mask)


# ─────────────────────── Triton kernel: SwiGLU activation ─────────────────── #
@triton.jit
def _swiglu_kernel(
    g1_ptr,      # [rows, 2*I] fp32 — GEMM1 output (gate‖up concatenated)
    c_ptr,       # [rows, I]   fp32 — SwiGLU output
    rows,
    i_size,      # I = intermediate_size
    sg10, sg11,  # strides for g1
    sc0, sc1,    # strides for c
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    SwiGLU: C[m, n] = X1[m, n] * silu(X2[m, n])
    where X1 = G1[:, :I]  (up projection)
    and   X2 = G1[:, I:]  (gate projection)
    silu(x) = x * sigmoid(x)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < rows
    mask_n = offs_n < i_size
    mask = mask_m[:, None] & mask_n[None, :]

    # X1 = first I columns (up), X2 = next I columns (gate)
    x1_ptrs = g1_ptr + offs_m[:, None] * sg10 + offs_n[None, :] * sg11
    x2_ptrs = g1_ptr + offs_m[:, None] * sg10 + (offs_n[None, :] + i_size) * sg11

    x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

    # SwiGLU = silu(gate) * up = [ x2 * sigmoid(x2) ] * x1
    silu_x2 = x2 * tl.sigmoid(x2)
    c = x1 * silu_x2

    c_ptrs = c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1
    tl.store(c_ptrs, c, mask=mask)


# ──────────────────── Python helpers: dequant + swiglu launchers ───────────── #
def _dequant_hidden_states(hidden_states, hidden_states_scale):
    """Launch Triton kernel to dequantize FP8 hidden states with block scales."""
    t_size, h_size = hidden_states.shape
    out = torch.empty((t_size, h_size), device=hidden_states.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N = 32, 128
    grid = (triton.cdiv(t_size, BLOCK_M), triton.cdiv(h_size, BLOCK_N))
    _dequant_hidden_fp8_kernel[grid](
        hidden_states, hidden_states_scale, out,
        t_size, h_size,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, SCALE_BLOCK=BLOCK_Q,
    )
    return out


def _swiglu(g1):
    """Launch Triton SwiGLU kernel on concatenated GEMM1 output."""
    rows = g1.shape[0]
    c = torch.empty((rows, INTERMEDIATE_SIZE), device=g1.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(rows, BLOCK_M), triton.cdiv(INTERMEDIATE_SIZE, BLOCK_N))
    _swiglu_kernel[grid](
        g1, c, rows, INTERMEDIATE_SIZE,
        g1.stride(0), g1.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return c


# ─────────────────── Weight dequantization (PyTorch, per-expert) ──────────── #
def _dequant_w13_local(w13_e, s13_e):
    """
    Dequant GEMM1 weights for one expert.
    w13_e: [2I, H] fp8,  s13_e: [(2I)/128, H/128] fp32
    Each 128×128 block shares one scale value.
    """
    n_out = 2 * INTERMEDIATE_SIZE // BLOCK_Q   # 32
    n_h   = HIDDEN_SIZE // BLOCK_Q             # 56
    w = w13_e.to(torch.float32).view(n_out, BLOCK_Q, n_h, BLOCK_Q)
    s = s13_e.to(torch.float32).view(n_out, 1, n_h, 1)
    return (w * s).reshape(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)


def _dequant_w2_local(w2_e, s2_e):
    """
    Dequant GEMM2 weights for one expert.
    w2_e: [H, I] fp8,  s2_e: [H/128, I/128] fp32
    """
    n_h = HIDDEN_SIZE // BLOCK_Q               # 56
    n_i = INTERMEDIATE_SIZE // BLOCK_Q          # 16
    w = w2_e.to(torch.float32).view(n_h, BLOCK_Q, n_i, BLOCK_Q)
    s = s2_e.to(torch.float32).view(n_h, 1, n_i, 1)
    return (w * s).reshape(HIDDEN_SIZE, INTERMEDIATE_SIZE)


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
    output: torch.Tensor,               # [T, H] bf16 — write in-place
):
    """
    Fused MoE layer (DeepSeek-V3 style) — DPS kernel.

    Pipeline:
      1. FP8 block-scale dequantization of hidden states
      2. DeepSeek-V3 no-aux routing (sigmoid → grouped topk → expert select)
      3. Per-expert: GEMM1 → SwiGLU → GEMM2 → weighted accumulation
      4. Write BF16 result into pre-allocated output tensor
    """
    # ── shape validation ───────────────────────────────────────────────────── #
    if routing_logits.shape[1] != NUM_EXPERTS:
        raise ValueError(f"Expected num_experts={NUM_EXPERTS}, got {routing_logits.shape[1]}")
    if hidden_states.shape[1] != HIDDEN_SIZE:
        raise ValueError(f"Expected hidden_size={HIDDEN_SIZE}, got {hidden_states.shape[1]}")
    if gemm1_weights.shape[0] != NUM_LOCAL_EXPERTS:
        raise ValueError(f"Expected num_local_experts={NUM_LOCAL_EXPERTS}, got {gemm1_weights.shape[0]}")

    t_size = routing_logits.shape[0]
    local_start = int(local_expert_offset)
    device = hidden_states.device

    # Ensure contiguity for Triton kernels
    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    routing_logits = routing_logits.contiguous()
    routing_bias = routing_bias.contiguous()

    # ── Stage 1: FP8 block-scale dequantization ───────────────────────────── #
    a = _dequant_hidden_states(hidden_states, hidden_states_scale)

    # ── Stage 2: DeepSeek-V3 no-aux routing ───────────────────────────────── #
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    # Sigmoid scores
    s = torch.sigmoid(logits)                                            # [T, E]
    s_with_bias = s + bias                                               # [T, E]

    # Group scoring: reshape to [T, N_GROUP, GROUP_SIZE], top-2 per group
    s_wb_grouped = s_with_bias.view(t_size, N_GROUP, GROUP_SIZE)         # [T, 8, 32]
    top2_vals = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2_vals.sum(dim=2)                                  # [T, 8]

    # Select top-4 groups
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1,
                           largest=True, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)        # [T, 8]
    group_mask.scatter_(1, group_idx, True)

    # Expand group mask and prune scores
    score_mask = (group_mask
                  .unsqueeze(2)
                  .expand(t_size, N_GROUP, GROUP_SIZE)
                  .reshape(t_size, NUM_EXPERTS))                         # [T, E]
    scores_pruned = s_with_bias.masked_fill(~score_mask, float("-inf"))

    # Global top-8 experts (within kept groups)
    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1,
                          largest=True, sorted=False).indices             # [T, 8]

    # Combination weights: normalize from s (without bias), then scale
    topk_s = torch.gather(s, 1, topk_idx)                               # [T, 8]
    topk_w = topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * float(routed_scaling_factor)                       # [T, 8]

    # ── Stage 3: Local expert compute ──────────────────────────────────────── #
    accum = torch.zeros((t_size, HIDDEN_SIZE), dtype=torch.float32, device=device)

    # Map topk indices to local expert indices
    local_idx = topk_idx - local_start                                   # [T, 8]
    valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)      # [T, 8]

    for le in range(NUM_LOCAL_EXPERTS):
        # Find (token, topk_pos) pairs routed to local expert `le`
        sel = valid_local & (local_idx == le)
        if not torch.any(sel):
            continue

        token_idx, topk_pos = torch.nonzero(sel, as_tuple=True)
        if token_idx.numel() == 0:
            continue

        # Gather token hidden states
        a_e = a.index_select(0, token_idx)                               # [Tk, H]

        # Dequantize expert weights
        w13_e = _dequant_w13_local(gemm1_weights[le],
                                   gemm1_weights_scale[le])              # [2I, H]
        w2_e  = _dequant_w2_local(gemm2_weights[le],
                                  gemm2_weights_scale[le])               # [H, I]

        # GEMM1: [Tk, H] × [H, 2I] = [Tk, 2I]
        g1 = torch.matmul(a_e, w13_e.t())

        # SwiGLU activation: silu(gate) * up
        c = _swiglu(g1)                                                  # [Tk, I]

        # GEMM2: [Tk, I] × [I, H] = [Tk, H]
        o = torch.matmul(c, w2_e.t())

        # Weighted accumulation
        w_tok = topk_w[token_idx, topk_pos].to(torch.float32)           # [Tk]
        accum.index_add_(0, token_idx, o * w_tok.unsqueeze(1))

    # ── Write result into pre-allocated DPS output ─────────────────────────── #
    output.copy_(accum.to(torch.bfloat16))
