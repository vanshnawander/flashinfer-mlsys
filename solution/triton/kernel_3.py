"""
Triton optimized implementation for the MLSys'26 fused_moe track:
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Submission-3 optimizations (targeting NVIDIA B200 Blackwell):
  1. Fused GEMM + FP8 dequant via Triton tl.dot (fixed scale indexing)
  2. Adaptive compute path: Triton GEMM for large batches, cuBLAS for small
  3. Triton SwiGLU kernel (fused silu + gate multiply)
  4. Pre-computed expert dispatch (single nonzero scan instead of per-expert)
  5. FP32 accumulation, BF16 output cast
  6. DPS: writes into pre-allocated output tensor

Key changes from submission-2:
  - Fixed fused GEMM scale loading to use proper stride indexing
  - Pre-compute dispatch table for all experts in one pass (avoids 32 nonzero calls)
  - Raised fused GEMM threshold to 32 for better accuracy on edge cases
  - Use torch.nn.functional.silu for small-batch SwiGLU fallback
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════ Geometry Constants ══════════════════════════════ #
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
BLOCK_Q = 128              # FP8 quantization block size
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = NUM_EXPERTS // N_GROUP   # 32


# ━━━━━━━━━━━ Triton Kernel: FP8 Dequant Hidden ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
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

    x_ptrs = x_ptr + offs_m[:, None] * sx0 + offs_n[None, :] * sx1
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    h_block = offs_n // SCALE_BLOCK
    s_ptrs = s_ptr + h_block[None, :] * ss0 + offs_m[:, None] * ss1
    s = tl.load(s_ptrs, mask=mask, other=0.0).to(tl.float32)

    tl.store(o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1, x * s, mask=mask)


# ━━━━━━━━━━━ Triton Kernel: SwiGLU ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
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


# ━━━━━━━━━━━ Triton Kernel: Fused GEMM + FP8 Dequant (fixed) ━━━━━━━━━━━━━ #
@triton.jit
def _gemm_fp8_dequant_kernel(
    a_ptr,        # [M, K] fp32 — activations
    w_ptr,        # [N, K] fp8  — weight (compute A @ W^T)
    s_ptr,        # [N/128, K/128] fp32 — block scales
    o_ptr,        # [M, N] fp32 — output
    M, N, K,
    sa0, sa1,     # A strides
    sw0, sw1,     # W strides
    ss0, ss1,     # scale strides [row=N/128, col=K/128]
    so0, so1,     # output strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused GEMM with on-the-fly FP8 block-scale weight dequantization.
    O[m, n] = sum_k A[m, k] * (W[n, k] * S[n // 128, k // 128])

    BLOCK_K must be 128 to align with quantization block size.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Scale block indices for N dimension (constant across K loop)
    n_block_idx = offs_n // 128  # [BLOCK_N]

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_block_idx = k_start // 128  # scalar — one K block per iteration

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )

        # Load W tile (FP8): [BLOCK_N, BLOCK_K]
        w_tile = tl.load(
            w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)

        # Load per-block scales: s[n_block, k_block] — 1D vector of BLOCK_N
        s_tile = tl.load(
            s_ptr + n_block_idx * ss0 + k_block_idx * ss1,
            mask=mask_n, other=1.0
        ).to(tl.float32)

        # Dequant weight: broadcast scale [BLOCK_N, 1] over [BLOCK_N, BLOCK_K]
        w_dequant = w_tile * s_tile[:, None]

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        acc += tl.dot(a_tile, tl.trans(w_dequant))

    tl.store(
        o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1,
        acc, mask=mask_m[:, None] & mask_n[None, :]
    )


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
    """SwiGLU fallback using PyTorch — lower launch overhead for tiny batches."""
    x1 = g1[:, :INTERMEDIATE_SIZE]
    x2 = g1[:, INTERMEDIATE_SIZE:]
    return (x1 * torch.nn.functional.silu(x2)).to(torch.float32)


def _gemm_with_fp8_dequant(a, w_fp8, w_scale, M, N, K):
    """
    Fused GEMM + FP8 dequant: computes A @ dequant(W)^T
    a: [M, K] fp32,  w_fp8: [N, K] fp8,  w_scale: [N/128, K/128] fp32
    Returns: [M, N] fp32
    """
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    # BLOCK_K=128 must align with BLOCK_Q for correct scale block indexing
    BM, BN, BK = 64, 64, 128
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm_fp8_dequant_kernel[grid](
        a, w_fp8, w_scale, out,
        M, N, K,
        a.stride(0), a.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return out


# ━━━━━━━━━━━ Weight Dequant Fallback (cuBLAS path) ━━━━━━━━━━━━━━━━━━━━━━━━ #
def _dequant_w13_local(w13_e, s13_e):
    n_out = 2 * INTERMEDIATE_SIZE // BLOCK_Q
    n_h = HIDDEN_SIZE // BLOCK_Q
    w = w13_e.to(torch.float32).view(n_out, BLOCK_Q, n_h, BLOCK_Q)
    s = s13_e.to(torch.float32).view(n_out, 1, n_h, 1)
    return (w * s).reshape(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)


def _dequant_w2_local(w2_e, s2_e):
    n_h = HIDDEN_SIZE // BLOCK_Q
    n_i = INTERMEDIATE_SIZE // BLOCK_Q
    w = w2_e.to(torch.float32).view(n_h, BLOCK_Q, n_i, BLOCK_Q)
    s = s2_e.to(torch.float32).view(n_h, 1, n_i, 1)
    return (w * s).reshape(HIDDEN_SIZE, INTERMEDIATE_SIZE)


# Adaptive threshold: use fused Triton GEMM above this token count
FUSED_GEMM_TOKEN_THRESHOLD = 32


# ═══════════════════════════════ MAIN KERNEL ══════════════════════════════════ #
@torch.no_grad()
def kernel(
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
    output: torch.Tensor,               # [T, H] bf16 — DPS output
):
    """
    Optimized fused MoE layer (DeepSeek-V3 style) — DPS kernel.

    Submission-3 optimizations:
      1. Fused GEMM+FP8 dequant (fixed scale indexing, tl.dot)
      2. Pre-computed dispatch table (single scan for all experts)
      3. Adaptive path: fused Triton GEMM for Tk≥32, cuBLAS for Tk<32
      4. PyTorch SwiGLU fallback for tiny batches
      5. FP32 accumulation throughout
    """
    t_size = routing_logits.shape[0]
    local_start = int(local_expert_offset)
    device = hidden_states.device

    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()

    # ── Stage 1: FP8 dequant hidden states ─────────────────────────────────── #
    a = _dequant_hidden_states(hidden_states, hidden_states_scale)

    # ── Stage 2: DeepSeek-V3 no-aux routing ────────────────────────────────── #
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    s_wb_grouped = s_with_bias.view(t_size, N_GROUP, GROUP_SIZE)
    top2_vals = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2_vals.sum(dim=2)

    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1,
                           largest=True, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)

    score_mask = (group_mask.unsqueeze(2)
                  .expand(t_size, N_GROUP, GROUP_SIZE)
                  .reshape(t_size, NUM_EXPERTS))
    scores_pruned = s_with_bias.masked_fill(~score_mask, float("-inf"))

    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1,
                          largest=True, sorted=False).indices

    topk_s = torch.gather(s, 1, topk_idx)
    topk_w = topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * float(routed_scaling_factor)

    # ── Stage 3: Pre-compute dispatch table ────────────────────────────────── #
    # Map topk indices to local expert IDs
    local_idx = topk_idx - local_start                                 # [T, 8]
    valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)    # [T, 8]

    # Pre-compute which experts are active and their token lists
    # This replaces 32 per-expert torch.nonzero calls with one scan
    expert_token_lists = [None] * NUM_LOCAL_EXPERTS
    expert_topk_lists = [None] * NUM_LOCAL_EXPERTS

    if torch.any(valid_local):
        # Flatten valid entries: (token_idx, topk_pos) for all valid local experts
        all_valid_idx = torch.nonzero(valid_local, as_tuple=False)     # [N_valid, 2]
        if all_valid_idx.numel() > 0:
            flat_token_idx = all_valid_idx[:, 0]
            flat_topk_pos = all_valid_idx[:, 1]
            flat_expert_id = local_idx[flat_token_idx, flat_topk_pos]  # local expert id

            # Sort by expert for grouped processing
            sort_order = torch.argsort(flat_expert_id, stable=True)
            sorted_expert_id = flat_expert_id[sort_order]
            sorted_token_idx = flat_token_idx[sort_order]
            sorted_topk_pos = flat_topk_pos[sort_order]

            # Find boundaries of each expert group
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
                start = end

    # ── Stage 4: Expert compute with adaptive GEMM path ────────────────────── #
    accum = torch.zeros((t_size, HIDDEN_SIZE), dtype=torch.float32, device=device)

    for le in range(NUM_LOCAL_EXPERTS):
        token_idx = expert_token_lists[le]
        if token_idx is None:
            continue

        topk_pos = expert_topk_lists[le]
        Tk = token_idx.numel()
        a_e = a.index_select(0, token_idx)                             # [Tk, H]

        # ── Adaptive GEMM1 ──
        if Tk >= FUSED_GEMM_TOKEN_THRESHOLD:
            g1 = _gemm_with_fp8_dequant(
                a_e, gemm1_weights[le], gemm1_weights_scale[le],
                Tk, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE
            )
        else:
            w13_e = _dequant_w13_local(gemm1_weights[le], gemm1_weights_scale[le])
            g1 = torch.matmul(a_e, w13_e.t())

        # ── SwiGLU activation ──
        if Tk >= FUSED_GEMM_TOKEN_THRESHOLD:
            c = _swiglu(g1)
        else:
            c = _swiglu_torch(g1)

        # ── Adaptive GEMM2 ──
        if Tk >= FUSED_GEMM_TOKEN_THRESHOLD:
            o = _gemm_with_fp8_dequant(
                c, gemm2_weights[le], gemm2_weights_scale[le],
                Tk, HIDDEN_SIZE, INTERMEDIATE_SIZE
            )
        else:
            w2_e = _dequant_w2_local(gemm2_weights[le], gemm2_weights_scale[le])
            o = torch.matmul(c, w2_e.t())

        # ── Weighted accumulation ──
        w_tok = topk_w[token_idx, topk_pos].to(torch.float32)
        accum.index_add_(0, token_idx, o * w_tok.unsqueeze(1))

    # ── Write BF16 result into DPS output ──────────────────────────────────── #
    output.copy_(accum.to(torch.bfloat16))
