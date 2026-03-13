"""
Triton optimized MoE kernel — Submission 9
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Sub-9 = BF16 tensor cores + factored block-scale application

  MATHEMATICAL INSIGHT:
    FP8 E4M3 has 4 significant bits. BF16 has 8. FP8→BF16 is LOSSLESS.
    Instead of dequant(FP8→FP32) then truncate(FP32→BF16) which LOSES
    scale precision, we keep FP8 raw values, promote to BF16 losslessly,
    do BF16 dot (tensor cores), then apply scales in FP32 post-dot.

  RESULT:
    - BF16 tensor core speed (~2× TFLOPS vs TF32)
    - ZERO precision loss (mathematically equivalent to FP32 path)
    - 4× less gather bandwidth (FP8 vs FP32 tokens)
    - Eliminates separate dequant kernel entirely

  ARCHITECTURE (per expert):
    GEMM1+SwiGLU fused kernel:
      For each (m_tile, n_tile) and each k_block:
        raw_dot_gate = dot_bf16(a_fp8_as_bf16, w_gate_fp8_as_bf16)  -- BF16 TC
        raw_dot_up   = dot_bf16(a_fp8_as_bf16, w_up_fp8_as_bf16)    -- BF16 TC
        gate_acc += raw_dot_gate * a_scale[m, k_blk] * w_scale[n_blk, k_blk]  -- FP32
        up_acc   += raw_dot_up   * a_scale[m, k_blk] * w_scale[n_up_blk, k_blk]  -- FP32
      SwiGLU: store gate_acc * silu(up_acc)

    GEMM2 kernel:
      Input c is FP32 from SwiGLU. INTER=2048 is small (K dim).
      We keep c in FP32, dequant w2 to FP32, use FP32 dot (TF32 TC).
      (BF16 trick doesn't apply here: c is FP32, not from FP8)
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

FUSED_GEMM_THRESHOLD = 32


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Fused GEMM1 + SwiGLU with factored block-scale BF16 dot                     #
#                                                                               #
#  For each tile (m_block, n_block) over K-blocks:                              #
#    1. Load a_fp8 tile → promote to BF16 (LOSSLESS: 4 bits → 8 bits)          #
#    2. Load w_gate_fp8 and w_up_fp8 → promote to BF16 (LOSSLESS)              #
#    3. BF16 × BF16 dot → raw_dot in FP32 (uses BF16 tensor cores)             #
#    4. Load a_scale (per-token, per-K-block) and w_scale (per-N-block, K-blk)  #
#    5. acc += raw_dot * a_scale[:, None] * w_scale (FP32 multiply)             #
#    6. After all K-blocks: SwiGLU epilogue in FP32                             #
#                                                                               #
#  This avoids ALL precision-losing truncation while using BF16 tensor cores.   #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _fused_gemm1_swiglu_bf16_kernel(
    # A (hidden states): FP8, shape (M, K) where K=HIDDEN_SIZE
    a_ptr,
    # A scales: FP32, shape (num_h_blocks, M) = (K//128, M)
    # Indexed as: a_scale[k_block, m] with strides (sas0, sas1)
    a_scale_ptr,
    # W (gemm1 weight): FP8, shape (2*N_HALF, K) where N_HALF=INTERMEDIATE_SIZE
    # Rows 0:N_HALF = gate, rows N_HALF:2*N_HALF = up
    w_ptr,
    # W scales: FP32, shape (2*N_HALF//128, K//128) = (32, 56)
    w_scale_ptr,
    # Output: FP32, shape (M, N_HALF)
    c_ptr,
    M, N_HALF, K,
    # Strides
    sa0, sa1,       # A: (M, K)
    sas0, sas1,     # A_scale: (K//128, M) — NOTE: h_block is dim0, token is dim1
    sw0, sw1,       # W: (2*N_HALF, K)
    sws0, sws1,     # W_scale: (2*N_HALF//128, K//128)
    sc0, sc1,       # C: (M, N_HALF)
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

    # W scale block indices (fixed for this tile since BLOCK_N=128=BLOCK_Q)
    # Gate rows are offs_n (0:INTER), up rows are offs_n + N_HALF
    n_scale_gate = pid_n                     # offs_n[0] // 128
    n_scale_up = pid_n + (N_HALF // 128)     # (offs_n[0] + N_HALF) // 128

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_blk = k_start // BLOCK_K    # BLOCK_K=128=BLOCK_Q, so this is the scale index

        # ── Load A FP8 → BF16 (LOSSLESS: E4M3 has 4 sig bits, BF16 has 8) ──
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)  # (BLOCK_M, BLOCK_K)

        # ── Load W_gate FP8 → BF16 (LOSSLESS) ──
        w_gate = tl.load(
            w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)  # (BLOCK_N, BLOCK_K)

        # ── Load W_up FP8 → BF16 (LOSSLESS) ──
        w_up = tl.load(
            w_ptr + (offs_n[:, None] + N_HALF) * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)  # (BLOCK_N, BLOCK_K)

        # ── BF16 tensor core dot: (M,K) @ (K,N) ──
        # Need (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        # w is (BLOCK_N, BLOCK_K), so we need its transpose
        # Use transposed load instead of tl.trans to avoid bad codegen:
        # Load w in (K, N) layout by swapping index roles
        # Actually, re-load in transposed order for clean codegen:

        w_gate_t = tl.load(
            w_ptr + offs_k[:, None] * sw1 + offs_n[None, :] * sw0,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0
        ).to(tl.bfloat16)  # (BLOCK_K, BLOCK_N)

        w_up_t = tl.load(
            w_ptr + offs_k[:, None] * sw1 + (offs_n[None, :] + N_HALF) * sw0,
            mask=mask_k[:, None] & mask_n[None, :], other=0.0
        ).to(tl.bfloat16)  # (BLOCK_K, BLOCK_N)

        # BF16 dot: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) → FP32
        raw_gate = tl.dot(a_tile, w_gate_t)   # (BLOCK_M, BLOCK_N) FP32
        raw_up = tl.dot(a_tile, w_up_t)       # (BLOCK_M, BLOCK_N) FP32

        # ── Load scales (FP32) ──
        # A scale: per-token, per-K-block → vector of BLOCK_M
        a_s = tl.load(
            a_scale_ptr + k_blk * sas0 + offs_m * sas1,
            mask=mask_m, other=1.0
        ).to(tl.float32)  # (BLOCK_M,)

        # W scale: per-N-block, per-K-block → scalar (since BLOCK_N=128=BLOCK_Q)
        ws_gate = tl.load(w_scale_ptr + n_scale_gate * sws0 + k_blk * sws1).to(tl.float32)
        ws_up = tl.load(w_scale_ptr + n_scale_up * sws0 + k_blk * sws1).to(tl.float32)

        # ── Apply scales in FP32 (ZERO precision loss) ──
        # Combined scale: a_s[m] * ws_gate (broadcast over N)
        combined_gate = a_s[:, None] * ws_gate   # (BLOCK_M, 1) * scalar
        combined_up = a_s[:, None] * ws_up

        gate_acc += raw_gate * combined_gate
        up_acc += raw_up * combined_up

    # ── SwiGLU epilogue (all FP32) ──
    result = gate_acc * (up_acc * tl.sigmoid(up_acc))

    tl.store(
        c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1,
        result,
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  GEMM2 kernel: SwiGLU output (FP32) × W2 (FP8 dequanted)                     #
#                                                                               #
#  Input C is FP32 from SwiGLU (NOT from FP8), so BF16 trick does NOT apply.    #
#  K=INTERMEDIATE_SIZE=2048 is small, so TF32 is fine.                          #
#  We keep the proven Sub-7 approach: dequant W to FP32, FP32 dot.              #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _gemm2_fp32_kernel(
    c_ptr, w_ptr, s_ptr, o_ptr,
    M, N, K,
    sc0, sc1, sw0, sw1, ss0, ss1, so0, so1,
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

        # C (SwiGLU output) is FP32
        c_tile = tl.load(
            c_ptr + offs_m[:, None] * sc0 + offs_k[None, :] * sc1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )  # FP32

        # W2 FP8 → FP32
        w_tile = tl.load(
            w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)

        s_tile = tl.load(
            s_ptr + n_block_idx * ss0 + k_block_idx * ss1,
            mask=mask_n, other=1.0
        ).to(tl.float32)

        w_dequant = w_tile * s_tile[:, None]  # FP32
        acc += tl.dot(c_tile, tl.trans(w_dequant))  # TF32 tensor cores

    tl.store(
        o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1,
        acc, mask=mask_m[:, None] & mask_n[None, :]
    )


# ═══════════════════════════ Python Launchers ═════════════════════════════════ #
def _launch_fused_gemm1_swiglu(a_fp8, a_scale, w_fp8, w_scale, Tk, c_out):
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(Tk, BM), triton.cdiv(INTERMEDIATE_SIZE, BN))
    _fused_gemm1_swiglu_bf16_kernel[grid](
        a_fp8, a_scale, w_fp8, w_scale, c_out,
        Tk, INTERMEDIATE_SIZE, HIDDEN_SIZE,
        a_fp8.stride(0), a_fp8.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        c_out.stride(0), c_out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_stages=3, num_warps=4,
    )


def _launch_gemm2(c_e, w_fp8, w_scale, Tk, o_out):
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(Tk, BM), triton.cdiv(HIDDEN_SIZE, BN))
    _gemm2_fp32_kernel[grid](
        c_e, w_fp8, w_scale, o_out,
        Tk, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        c_e.stride(0), c_e.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        o_out.stride(0), o_out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_stages=3, num_warps=4,
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
    """Full FP32 dequant for cuBLAS fallback path only."""
    t_size, h_size = hidden_states.shape
    nb_h = h_size // BLOCK_Q
    x = hidden_states.to(torch.float32).view(t_size, nb_h, BLOCK_Q)
    # Scale shape: (nb_h, t_size) → need (t_size, nb_h, 1)
    s = hidden_states_scale.to(torch.float32).t().unsqueeze(2)  # (t_size, nb_h, 1)
    return (x * s).reshape(t_size, h_size)


# ═══════════════════════════════ MAIN KERNEL ══════════════════════════════════ #
@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,       # (T, 7168) FP8
    hidden_states_scale: torch.Tensor,  # (56, T) FP32 — h_block × token
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

    # ── Stage 1: Routing (fully vectorized, no change) ─────────────────── #
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

    # ── Stage 2: Build sorted-by-expert token layout ───────────────────── #
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

    # ── Stage 3: Bulk gather ───────────────────────────────────────────── #
    # FP8 gather: 1 byte/elem (vs 4 bytes for FP32) → 4× less bandwidth
    N_valid = sorted_token_idx.numel()
    use_bulk = (N_valid >= 64)

    if use_bulk:
        # Gather FP8 hidden states (4× cheaper than FP32)
        sorted_a_fp8 = hidden_states.index_select(0, sorted_token_idx)
        # Gather scales: hidden_states_scale is (num_h_blocks, T)
        # We need columns corresponding to sorted_token_idx
        sorted_a_scale = hidden_states_scale.index_select(1, sorted_token_idx)
        sorted_w = topk_w[sorted_token_idx, sorted_topk_pos].to(torch.float32)

    # For cuBLAS fallback, we need FP32 dequanted hidden states (lazy)
    a_fp32_cache = None

    # Pre-allocate scratch buffers
    max_tk = int(counts.max().item())
    c_buf = torch.empty((max_tk, INTERMEDIATE_SIZE), device=device, dtype=torch.float32)
    o_buf = torch.empty((max_tk, HIDDEN_SIZE), device=device, dtype=torch.float32)

    # ── Stage 4: Per-expert compute ────────────────────────────────────── #
    start = 0
    for i in range(unique_experts.numel()):
        le = unique_experts[i].item()
        end = boundaries[i].item()
        Tk = end - start

        t_idx = sorted_token_idx[start:end]

        if use_bulk:
            w_e = sorted_w[start:end]
        else:
            w_e = topk_w[t_idx, sorted_topk_pos[start:end]].to(torch.float32)

        if Tk >= FUSED_GEMM_THRESHOLD:
            # ── Triton path: BF16 tensor cores with factored scales ──

            # Get FP8 tokens and their scales for this expert
            if use_bulk:
                a_e_fp8 = sorted_a_fp8[start:end]         # (Tk, 7168) FP8 contiguous
                a_e_scale = sorted_a_scale[:, start:end]   # (56, Tk) FP32
            else:
                a_e_fp8 = hidden_states.index_select(0, t_idx)
                a_e_scale = hidden_states_scale.index_select(1, t_idx)

            # Fused GEMM1 + SwiGLU (BF16 tensor cores)
            c_view = c_buf[:Tk]
            _launch_fused_gemm1_swiglu(
                a_e_fp8, a_e_scale,
                gemm1_weights[le], gemm1_weights_scale[le],
                Tk, c_view
            )

            # GEMM2 (SwiGLU output is FP32, so use FP32/TF32 path)
            o_view = o_buf[:Tk]
            _launch_gemm2(
                c_view, gemm2_weights[le], gemm2_weights_scale[le],
                Tk, o_view
            )
        else:
            # ── cuBLAS fallback for tiny experts ──
            # Dequant hidden states to FP32 (only on first use)
            if a_fp32_cache is None:
                a_fp32_cache = _dequant_hidden_fp32(hidden_states, hidden_states_scale)

            a_e = a_fp32_cache.index_select(0, t_idx)

            w13_e = _dequant_weight(gemm1_weights[le], gemm1_weights_scale[le],
                                    2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
            g1 = torch.matmul(a_e, w13_e.t())
            c_result = _swiglu_torch(g1)

            w2_e = _dequant_weight(gemm2_weights[le], gemm2_weights_scale[le],
                                   HIDDEN_SIZE, INTERMEDIATE_SIZE)
            o_view = torch.matmul(c_result, w2_e.t())

        # ── Weighted scatter-add ──
        accum.index_add_(0, t_idx, o_view * w_e.unsqueeze(1))

        start = end

    # ── Write BF16 ─────────────────────────────────────────────────────── #
    output.copy_(accum.to(torch.bfloat16))