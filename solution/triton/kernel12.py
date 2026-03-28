"""
Triton optimized MoE kernel — Submission 12 (conservative)
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Sub-12 = EXACT Sub-9 GEMM precision + route-weight fusion only

  ROOT CAUSE OF SUB-12-old FAILURE:
    BF16 tl.dot with tl.trans() produces wrong results on certain 
    architectures. The tl.trans "layout renaming" plus convertLayout
    ops can silently corrupt data with BF16 operands.
    
    Sub-7/Sub-9 worked because they used FP32 operands with tl.trans —
    the FP32 path exercises a different (working) code path in the
    Triton compiler's layout conversion.

  APPROACH:
    - GEMM1+SwiGLU: FP8→FP32 dequant, FP32 dot with tl.trans (Sub-9 exact)
    - GEMM2: FP32 dot with tl.trans (Sub-7 exact)  
    - NEW: route-weight fusion in GEMM2 epilogue (pure FP32, no precision change)
    - Fused GEMM1+SwiGLU (shared A loads, no intermediate buf)
    
  WHAT WE DO NOT CHANGE FROM SUB-9:
    ✓ FP32 hidden state dequant
    ✓ FP32 weight dequant inside GEMM loops
    ✓ FP32 operands to tl.dot everywhere
    ✓ FP32 SwiGLU output
    ✓ Coalesced (N,K) weight loads + tl.trans
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
GROUP_SIZE = NUM_EXPERTS // N_GROUP

FUSED_GEMM_THRESHOLD = 32


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  FUSED GEMM1 + SwiGLU — FP32 throughout (Sub-9 precision, proven correct)    #
#  Optimization: A tile loaded once, used for both gate and up projections     #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _fused_gemm1_swiglu_kernel(
    a_ptr,           # (M, K) FP32 dequanted hidden states
    w_ptr,           # (2*N_HALF, K) FP8
    w_scale_ptr,     # (2*N_HALF//128, K//128) FP32
    c_ptr,           # (M, N_HALF) FP32 output
    M, N_HALF, K,
    sa0, sa1,
    sw0, sw1,
    sws0, sws1,
    sc0, sc1,
    N_HALF_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N_HALF

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_scale_gate = pid_n
    n_scale_up = pid_n + N_HALF_BLOCKS

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_blk = k_start // BLOCK_K

        # A: already FP32 (from dequant kernel)
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )  # (BLOCK_M, BLOCK_K) FP32

        # W_gate: coalesced FP8 load → FP32 dequant
        w_gate = tl.load(
            w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)  # (BLOCK_N, BLOCK_K)

        # W_up: offset by N_HALF rows
        w_up = tl.load(
            w_ptr + (offs_n[:, None] + N_HALF) * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)  # (BLOCK_N, BLOCK_K)

        # Scalar weight scales
        ws_gate = tl.load(w_scale_ptr + n_scale_gate * sws0 + k_blk * sws1).to(tl.float32)
        ws_up = tl.load(w_scale_ptr + n_scale_up * sws0 + k_blk * sws1).to(tl.float32)

        # Dequant weights in FP32
        w_gate_dq = w_gate * ws_gate   # (BLOCK_N, BLOCK_K)
        w_up_dq = w_up * ws_up         # (BLOCK_N, BLOCK_K)

        # FP32 dot with tl.trans (proven working on B200 in Sub-7/Sub-9)
        gate_acc += tl.dot(a_tile, tl.trans(w_gate_dq))
        up_acc += tl.dot(a_tile, tl.trans(w_up_dq))

    # SwiGLU epilogue (all FP32)
    result = gate_acc * (up_acc * tl.sigmoid(up_acc))

    tl.store(
        c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1,
        result,
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  GEMM2 + fused route-weight epilogue — FP32 throughout                       #
#  Only change from Sub-9: route_w multiply fused before store                 #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _gemm2_weighted_kernel(
    c_ptr,           # (M, K) FP32 — SwiGLU output
    w_ptr,           # (N, K) FP8
    s_ptr,           # (N//128, K//128) FP32
    route_w_ptr,     # (M,) FP32 — routing weights
    o_ptr,           # (M, N) FP32 output
    M, N, K,
    sc0, sc1,
    sw0, sw1,
    ss0, ss1,
    so0, so1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_block_idx = pid_n

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_block_idx = k_start // BLOCK_K

        # C: FP32 from SwiGLU
        c_tile = tl.load(
            c_ptr + offs_m[:, None] * sc0 + offs_k[None, :] * sc1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )

        # W2: coalesced FP8 load → FP32 dequant
        w_tile = tl.load(
            w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)

        s_val = tl.load(s_ptr + n_block_idx * ss0 + k_block_idx * ss1).to(tl.float32)
        w_dequant = w_tile * s_val

        # FP32 dot with tl.trans (identical to Sub-7/Sub-9, proven correct)
        acc += tl.dot(c_tile, tl.trans(w_dequant))

    # NEW: fused route-weight multiply (pure FP32 in-register, zero precision change)
    route_w = tl.load(
        route_w_ptr + offs_m, mask=mask_m, other=0.0
    ).to(tl.float32)
    acc = acc * route_w[:, None]

    tl.store(
        o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1,
        acc,
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  FP8 Hidden State Dequant → FP32 (identical to Sub-9)                        #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
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


# ═══════════════════════════ Python Launchers ═════════════════════════════════ #
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


def _launch_fused_gemm1_swiglu(a, w_fp8, w_scale, Tk, c_out):
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(Tk, BM), triton.cdiv(INTERMEDIATE_SIZE, BN))
    _fused_gemm1_swiglu_kernel[grid](
        a, w_fp8, w_scale, c_out,
        Tk, INTERMEDIATE_SIZE, HIDDEN_SIZE,
        a.stride(0), a.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        c_out.stride(0), c_out.stride(1),
        N_HALF_BLOCKS=INTERMEDIATE_SIZE // 128,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_stages=3, num_warps=4,
    )


def _launch_gemm2_weighted(c_e, w_fp8, w_scale, route_w, Tk, o_out):
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(Tk, BM), triton.cdiv(HIDDEN_SIZE, BN))
    _gemm2_weighted_kernel[grid](
        c_e, w_fp8, w_scale, route_w, o_out,
        Tk, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        c_e.stride(0), c_e.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        o_out.stride(0), o_out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_stages=4, num_warps=4,
    )


def _dequant_weight(w_fp8, scale, out_dim, in_dim):
    nb_out = out_dim // BLOCK_Q
    nb_in = in_dim // BLOCK_Q
    w = w_fp8.to(torch.float32).view(nb_out, BLOCK_Q, nb_in, BLOCK_Q)
    s = scale.to(torch.float32).view(nb_out, 1, nb_in, 1)
    return (w * s).reshape(out_dim, in_dim)


def _swiglu_torch(g1):
    x1 = g1[:, :INTERMEDIATE_SIZE]
    x2 = g1[:, INTERMEDIATE_SIZE:]
    return (x1 * torch.nn.functional.silu(x2)).to(torch.float32)


# ═══════════════════════════════ MAIN KERNEL ══════════════════════════════════ #
@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,
):
    t_size = routing_logits.shape[0]
    local_start = int(local_expert_offset)
    device = hidden_states.device

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()

    # ── FP32 dequant (identical to Sub-9) ──
    a = _dequant_hidden_states(hidden_states, hidden_states_scale)

    # ── Routing (identical to Sub-9) ──
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).view(-1)

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

    score_mask = group_mask.unsqueeze(2).expand(
        t_size, N_GROUP, GROUP_SIZE).reshape(t_size, NUM_EXPERTS)
    scores_pruned = s_with_bias.masked_fill(~score_mask, float("-inf"))

    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1,
                          largest=True, sorted=False).indices

    topk_s = torch.gather(s, 1, topk_idx)
    topk_w = topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * float(routed_scaling_factor)

    # ── Dispatch ──
    local_idx = topk_idx - local_start
    valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)

    accum = torch.zeros((t_size, HIDDEN_SIZE), dtype=torch.float32, device=device)

    all_valid_idx = torch.nonzero(valid_local, as_tuple=False)
    if all_valid_idx.numel() == 0:
        output.copy_(accum.to(torch.bfloat16))
        return

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

    # ── Bulk gather (FP32 tokens) ──
    N_valid = sorted_token_idx.numel()
    use_bulk = (N_valid >= 64)

    if use_bulk:
        sorted_a = a.index_select(0, sorted_token_idx)
        sorted_w = topk_w[sorted_token_idx, sorted_topk_pos].to(torch.float32)

    # Pre-allocate scratch
    max_tk = int(counts.max().item())
    c_buf = torch.empty((max_tk, INTERMEDIATE_SIZE), device=device, dtype=torch.float32)
    o_buf = torch.empty((max_tk, HIDDEN_SIZE), device=device, dtype=torch.float32)

    # ── Per-expert compute ──
    start = 0
    for i in range(unique_experts.numel()):
        le = unique_experts[i].item()
        end = boundaries[i].item()
        Tk = end - start

        t_idx = sorted_token_idx[start:end]

        if use_bulk:
            a_e = sorted_a[start:end]
            w_e = sorted_w[start:end]
        else:
            a_e = a.index_select(0, t_idx)
            w_e = topk_w[t_idx, sorted_topk_pos[start:end]].to(torch.float32)

        if Tk >= FUSED_GEMM_THRESHOLD:
            # Fused GEMM1+SwiGLU (saves 1 kernel launch + intermediate buffer)
            c_view = c_buf[:Tk]
            _launch_fused_gemm1_swiglu(
                a_e, gemm1_weights[le], gemm1_weights_scale[le],
                Tk, c_view
            )

            # GEMM2 with fused route-weight (saves 1 full bandwidth pass)
            o_view = o_buf[:Tk]
            _launch_gemm2_weighted(
                c_view, gemm2_weights[le], gemm2_weights_scale[le],
                w_e, Tk, o_view
            )

            # Route-weight already applied in GEMM2 epilogue
            accum.index_add_(0, t_idx, o_view)

        else:
            # cuBLAS fallback
            w13_e = _dequant_weight(gemm1_weights[le], gemm1_weights_scale[le],
                                    2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
            g1 = torch.matmul(a_e, w13_e.t())
            c_result = _swiglu_torch(g1)

            w2_e = _dequant_weight(gemm2_weights[le], gemm2_weights_scale[le],
                                   HIDDEN_SIZE, INTERMEDIATE_SIZE)
            o_result = torch.matmul(c_result, w2_e.t())

            accum.index_add_(0, t_idx, o_result * w_e.unsqueeze(1))

        start = end

    output.copy_(accum.to(torch.bfloat16))