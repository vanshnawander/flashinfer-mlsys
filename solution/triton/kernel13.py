"""
Triton optimized MoE kernel — Submission 12-final
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Sub-13 = EXACT Sub-9 + 3-line GEMM2 epilogue change

  ROOT CAUSE OF PREVIOUS FAILURES:
    Fused GEMM1+SwiGLU with FP32 inputs: A tile = 64×128×4 = 32KB/stage
    × 3 stages = 96KB just for A. Total SMEM ~232KB > B200 limit ~228KB.
    
    GEMM2 with num_stages=4: C tile = 32KB/stage × 4 = 128KB, plus W
    tiles, total ~240KB > limit.

  FIX: Use Sub-9 EXACTLY (FP8 A tiles = 8KB/stage, fits in SMEM)
       Add ONLY route-weight multiply to GEMM2 epilogue (3 lines)
       Keep num_stages=3 everywhere (proven safe on B200)

  SAFE CHANGE (the ONLY modification from Sub-9):
    GEMM2 kernel: before tl.store, multiply acc by route_w[m]
    This is 256 bytes of extra data (64 floats) loaded once per M-tile.
    SMEM impact: zero (scalar loads, not pipelined)
    Register impact: 1 extra vector of 64 floats = 256 bytes ≈ negligible
"""

import torch
import triton
import triton.language as tl

# ═══════════════════════════ Constants (identical to Sub-9) ═══════════════ #
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
#  FUSED GEMM1 + SwiGLU — BYTE-FOR-BYTE IDENTICAL TO SUB-9                    #
#  FP8 inputs, BF16 tensor cores, post-dot FP32 scales                         #
#  SMEM per stage: 8KB(A) + 16KB(Wg) + 16KB(Wu) = 40KB × 3 = 120KB → OK     #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _fused_gemm1_swiglu_kernel(
    a_ptr,           # (M, K) FP8
    a_scale_ptr,     # (K//128, M_stride_dim) FP32
    w_ptr,           # (2*N_HALF, K) FP8
    w_scale_ptr,     # (2*N_HALF//128, K//128) FP32
    c_ptr,           # (M, N_HALF) FP32 output
    M, N_HALF, K,
    sa0, sa1,
    sas0, sas1,
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

        # A: FP8 → BF16 (LOSSLESS)
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)

        # W_gate: FP8 → BF16 (LOSSLESS), coalesced load
        w_gate = tl.load(
            w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)

        # W_up: FP8 → BF16 (LOSSLESS), coalesced load
        w_up = tl.load(
            w_ptr + (offs_n[:, None] + N_HALF) * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)

        # BF16 tensor core dot
        raw_gate = tl.dot(a_tile, tl.trans(w_gate))
        raw_up = tl.dot(a_tile, tl.trans(w_up))

        # Post-dot FP32 scale application
        a_s = tl.load(
            a_scale_ptr + k_blk * sas0 + offs_m * sas1,
            mask=mask_m, other=1.0
        ).to(tl.float32)

        ws_gate = tl.load(w_scale_ptr + n_scale_gate * sws0 + k_blk * sws1).to(tl.float32)
        ws_up = tl.load(w_scale_ptr + n_scale_up * sws0 + k_blk * sws1).to(tl.float32)

        gate_acc += raw_gate * (a_s[:, None] * ws_gate)
        up_acc += raw_up * (a_s[:, None] * ws_up)

    # SwiGLU in FP32
    result = gate_acc * (up_acc * tl.sigmoid(up_acc))

    tl.store(
        c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1,
        result,
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  GEMM2 — Sub-9 kernel + 3-line route-weight fusion                           #
#  FP32 input (SwiGLU), FP32 dequant weights, TF32 tensor cores                #
#  num_stages=3 (proven safe — 48KB × 3 = 144KB < 228KB limit)                #
#                                                                               #
#  THE ONLY CHANGE FROM SUB-9: multiply acc by route_w before store            #
#  This adds 256 bytes of scalar loads per M-tile — zero SMEM impact           #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _gemm2_weighted_kernel(
    c_ptr,           # (M, K) FP32
    w_ptr,           # (N, K) FP8
    s_ptr,           # (N//128, K//128) FP32
    route_w_ptr,     # (M,) FP32 — routing weights
    o_ptr,           # (M, N) FP32
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
    n_block_idx = offs_n // 128

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_block_idx = k_start // 128

        c_tile = tl.load(
            c_ptr + offs_m[:, None] * sc0 + offs_k[None, :] * sc1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )

        w_tile = tl.load(
            w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)

        s_tile = tl.load(
            s_ptr + n_block_idx * ss0 + k_block_idx * ss1,
            mask=mask_n, other=1.0
        ).to(tl.float32)

        w_dequant = w_tile * s_tile[:, None]
        acc += tl.dot(c_tile, tl.trans(w_dequant))

    # ═══════════════════════════════════════════════════════════════════ #
    #  THIS IS THE ONLY NEW CODE vs Sub-9 (3 lines)                     #
    #  Multiply accumulator by routing weight before store               #
    #  Saves one full (Tk × 7168 × 4 byte) Python read-modify-write     #
    # ═══════════════════════════════════════════════════════════════════ #
    route_w = tl.load(
        route_w_ptr + offs_m, mask=mask_m, other=0.0
    ).to(tl.float32)
    acc = acc * route_w[:, None]

    tl.store(
        o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1,
        acc, mask=mask_m[:, None] & mask_n[None, :]
    )


# ═══════════════════════════ Launchers (identical to Sub-9 except GEMM2) ══ #
def _launch_fused_gemm1_swiglu(a_fp8, a_scale, w_fp8, w_scale, Tk, c_out):
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(Tk, BM), triton.cdiv(INTERMEDIATE_SIZE, BN))
    _fused_gemm1_swiglu_kernel[grid](
        a_fp8, a_scale, w_fp8, w_scale, c_out,
        Tk, INTERMEDIATE_SIZE, HIDDEN_SIZE,
        a_fp8.stride(0), a_fp8.stride(1),
        a_scale.stride(0), a_scale.stride(1),
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
        num_stages=3, num_warps=4,    # ← MUST be 3, not 4! (4 overflows SMEM)
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


def _dequant_hidden_fp32(hidden_states, hidden_states_scale):
    t_size, h_size = hidden_states.shape
    nb_h = h_size // BLOCK_Q
    x = hidden_states.to(torch.float32).view(t_size, nb_h, BLOCK_Q)
    s = hidden_states_scale.to(torch.float32).t().unsqueeze(2)
    return (x * s).reshape(t_size, h_size)


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

    # ── Routing (identical to Sub-9) ──────────────────────────────────── #
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

    # ── Dispatch (identical to Sub-9) ─────────────────────────────────── #
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

    # ── Bulk gather — FP8, identical to Sub-9 ─────────────────────────── #
    N_valid = sorted_token_idx.numel()
    use_bulk = (N_valid >= 64)

    if use_bulk:
        sorted_a_fp8 = hidden_states.index_select(0, sorted_token_idx)
        sorted_a_scale = hidden_states_scale.index_select(1, sorted_token_idx)
        sorted_w = topk_w[sorted_token_idx, sorted_topk_pos].to(torch.float32)

    # Lazy FP32 dequant for cuBLAS fallback
    a_fp32_cache = None

    # Pre-allocate scratch (identical to Sub-9)
    max_tk = int(counts.max().item())
    c_buf = torch.empty((max_tk, INTERMEDIATE_SIZE), device=device, dtype=torch.float32)
    o_buf = torch.empty((max_tk, HIDDEN_SIZE), device=device, dtype=torch.float32)

    # ── Per-expert compute ────────────────────────────────────────────── #
    start = 0
    for i in range(unique_experts.numel()):
        le = unique_experts[i].item()
        end = boundaries[i].item()
        Tk = end - start

        t_idx = sorted_token_idx[start:end]

        if Tk >= FUSED_GEMM_THRESHOLD:
            # Get FP8 tokens + scales (identical to Sub-9)
            if use_bulk:
                a_e_fp8 = sorted_a_fp8[start:end]
                a_e_scale = sorted_a_scale[:, start:end]
                w_e = sorted_w[start:end]
            else:
                a_e_fp8 = hidden_states.index_select(0, t_idx)
                a_e_scale = hidden_states_scale.index_select(1, t_idx)
                w_e = topk_w[t_idx, sorted_topk_pos[start:end]].to(torch.float32)

            # Fused GEMM1+SwiGLU (identical to Sub-9)
            c_view = c_buf[:Tk]
            _launch_fused_gemm1_swiglu(
                a_e_fp8, a_e_scale,
                gemm1_weights[le], gemm1_weights_scale[le],
                Tk, c_view
            )

            # GEMM2 with fused route-weight (THE ONLY CHANGE)
            o_view = o_buf[:Tk]
            _launch_gemm2_weighted(
                c_view, gemm2_weights[le], gemm2_weights_scale[le],
                w_e, Tk, o_view
            )

            # Route-weight already applied → just scatter-add
            accum.index_add_(0, t_idx, o_view)

        else:
            # cuBLAS fallback (identical to Sub-9)
            if a_fp32_cache is None:
                a_fp32_cache = _dequant_hidden_fp32(hidden_states, hidden_states_scale)

            a_e = a_fp32_cache.index_select(0, t_idx)
            w_e = topk_w[t_idx, sorted_topk_pos[start:end]].to(torch.float32)

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