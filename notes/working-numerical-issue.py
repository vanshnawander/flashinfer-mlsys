"""
Triton optimized MoE kernel — Submission 8 (fixed)
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Sub-8 = Sub-7 redesign: fix all runtime errors + real tensor core usage

  FIXED RUNTIME ERRORS:
    ✓ Removed persistent kernel (non-constexpr loop bounds crash Triton)
    ✓ Removed tl.trans() inside tl.dot() (broken codegen on Ampere+)
    ✓ Load weight in transposed (K,N) order → natural layout for tl.dot
    ✓ Pass real tensor strides (not hardcoded computed strides)
    ✓ Proper BF16 tensor-core dot (was FP32 → no tensor cores at all)

  PERFORMANCE OPTIMIZATIONS:
    1. BF16 tl.dot: dequant weights to BF16, dot uses BF16 tensor cores
       → ~2× TFLOPS vs FP32/TF32 path on B200
    2. Fused GEMM1+SwiGLU kernel: eliminates (Tk, 4096) intermediate
       buffer write+read, saves 1 kernel launch per expert
    3. Scalar scale load: BLOCK_N=BLOCK_K=128=BLOCK_Q → 1 scale per tile
    4. Dequant hidden states to BF16 (not FP32) → halves gather bandwidth
    5. Bulk index_select for sorted tokens → O(1) gathers
    6. Pre-allocated scratch buffers reused across experts
    7. Adaptive cuBLAS fallback for tiny experts (Tk < 32)
    8. B200 tuning: num_stages=3, num_warps=4 for 64×128 tiles
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


# ━━━━━━━━━━━ Triton: FP8 Hidden State Dequant → BF16 ━━━━━━━━━━━━━━━━━━━━━ #
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
    # Output BF16 to halve bandwidth for subsequent gathers
    tl.store(o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1,
             (x * s).to(tl.bfloat16), mask=mask)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Fused GEMM1 + SwiGLU kernel                                                #
#  - Computes both gate (rows 0:INTER) and up (rows INTER:2*INTER) of W       #
#  - Shares A tile loads between gate and up → halves A bandwidth              #
#  - SwiGLU applied in epilogue → eliminates (Tk, 4096) intermediate buffer   #
#  - Weight loaded in (K, N) order → natural layout for tl.dot, no tl.trans   #
#  - BF16 dot → uses BF16 tensor cores on B200                                #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _gemm1_swiglu_kernel(
    a_ptr, w_ptr, ws_ptr, c_ptr,
    M, N_HALF,  # N_HALF = INTERMEDIATE_SIZE
    K,          # K = HIDDEN_SIZE
    sa0, sa1,       # A strides (M, K) BF16
    sw0, sw1,       # W strides: (2*N_HALF, K) FP8, sw0=row(N), sw1=col(K)
    sws0, sws1,     # WS strides: (2*N_HALF//128, K//128)
    sc0, sc1,       # C strides (M, N_HALF) FP32
    N_HALF_BLOCKS: tl.constexpr,  # N_HALF // 128
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

    # Scale block indices (scalar since BLOCK_N = 128 = BLOCK_Q)
    n_scale_gate = pid_n
    n_scale_up = pid_n + N_HALF_BLOCKS  # offset into upper half of weight

    # N offsets for the "up" half of the weight matrix
    offs_n_up = offs_n + N_HALF

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_scale = k_start // BLOCK_K  # == k_block_idx since BLOCK_K=128=BLOCK_Q

        # ── Load A tile: (BLOCK_M, BLOCK_K) BF16 ──
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)

        # ── Load gate weight (K, N) order for natural tl.dot layout ──
        # W is stored (N, K). We load as (BLOCK_K, BLOCK_N) by swapping indices
        # This avoids tl.trans() which generates broken SMEM transpose code
        w_gate_t = tl.load(
            w_ptr + offs_n[None, :] * sw0 + offs_k[:, None] * sw1,
            mask=mask_n[None, :] & mask_k[:, None], other=0.0
        ).to(tl.float32)

        # ── Load up weight in same transposed order ──
        w_up_t = tl.load(
            w_ptr + offs_n_up[None, :] * sw0 + offs_k[:, None] * sw1,
            mask=mask_n[None, :] & mask_k[:, None], other=0.0
        ).to(tl.float32)

        # ── Scalar scale loads (1 scale per BLOCK_N×BLOCK_K tile) ──
        s_gate = tl.load(ws_ptr + n_scale_gate * sws0 + k_scale * sws1).to(tl.float32)
        s_up = tl.load(ws_ptr + n_scale_up * sws0 + k_scale * sws1).to(tl.float32)

        # ── Dequant to BF16 for tensor core dot ──
        w_gate_dq = (w_gate_t * s_gate).to(tl.bfloat16)   # (BLOCK_K, BLOCK_N)
        w_up_dq = (w_up_t * s_up).to(tl.bfloat16)         # (BLOCK_K, BLOCK_N)

        # ── BF16 × BF16 tensor core dot ──
        # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) → (BLOCK_M, BLOCK_N)
        gate_acc += tl.dot(a_tile, w_gate_dq)
        up_acc += tl.dot(a_tile, w_up_dq)

    # ── SwiGLU epilogue: x1 * silu(x2) = gate * (up * sigmoid(up)) ──
    result = gate_acc * (up_acc * tl.sigmoid(up_acc))

    # Store FP32
    tl.store(
        c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1,
        result,
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  GEMM2 kernel: FP8 weight dequant with BF16 tensor core dot                  #
#  - Same transposed-load approach (no tl.trans)                               #
#  - Scalar scale per tile                                                     #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _gemm2_kernel(
    c_ptr, w_ptr, ws_ptr, o_ptr,
    M, N, K,      # N=HIDDEN_SIZE, K=INTERMEDIATE_SIZE
    sc0, sc1,     # C input strides (M, K)
    sw0, sw1,     # W strides (N, K) FP8
    sws0, sws1,   # WS strides (N//128, K//128)
    so0, so1,     # output strides (M, N)
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

    n_scale = pid_n  # scalar since BLOCK_N=128=BLOCK_Q

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_scale = k_start // BLOCK_K

        # Load C (SwiGLU output) as BF16 for tensor core dot
        c_tile = tl.load(
            c_ptr + offs_m[:, None] * sc0 + offs_k[None, :] * sc1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)

        # Load weight in (K, N) order — no tl.trans needed
        w_t = tl.load(
            w_ptr + offs_n[None, :] * sw0 + offs_k[:, None] * sw1,
            mask=mask_n[None, :] & mask_k[:, None], other=0.0
        ).to(tl.float32)

        s = tl.load(ws_ptr + n_scale * sws0 + k_scale * sws1).to(tl.float32)

        w_dq = (w_t * s).to(tl.bfloat16)  # (BLOCK_K, BLOCK_N)

        # BF16 tensor core dot
        acc += tl.dot(c_tile, w_dq)

    tl.store(
        o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1,
        acc,
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ═══════════════════════════ Python Launchers ═════════════════════════════════ #
def _dequant_hidden_states(hidden_states, hidden_states_scale):
    t_size, h_size = hidden_states.shape
    out = torch.empty((t_size, h_size), device=hidden_states.device, dtype=torch.bfloat16)
    BM, BN = 64, 128
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


def _launch_gemm1_swiglu(a_e, w_fp8, w_scale, Tk, out):
    """Fused GEMM1 + SwiGLU: a_e @ W^T then SwiGLU → (Tk, INTER)"""
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(Tk, BM), triton.cdiv(INTERMEDIATE_SIZE, BN))
    _gemm1_swiglu_kernel[grid](
        a_e, w_fp8, w_scale, out,
        Tk, INTERMEDIATE_SIZE, HIDDEN_SIZE,
        a_e.stride(0), a_e.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0), out.stride(1),
        N_HALF_BLOCKS=INTERMEDIATE_SIZE // 128,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_stages=3, num_warps=4,
    )


def _launch_gemm2(c_e, w_fp8, w_scale, Tk, out):
    """GEMM2: c_e @ W^T → (Tk, HIDDEN)"""
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(Tk, BM), triton.cdiv(HIDDEN_SIZE, BN))
    _gemm2_kernel[grid](
        c_e, w_fp8, w_scale, out,
        Tk, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        c_e.stride(0), c_e.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_stages=3, num_warps=4,
    )


def _dequant_weight(w_fp8, scale, out_dim, in_dim):
    """Fallback: full weight dequant for cuBLAS path (small Tk only)."""
    nb_out = out_dim // BLOCK_Q
    nb_in = in_dim // BLOCK_Q
    w = w_fp8.to(torch.float32).view(nb_out, BLOCK_Q, nb_in, BLOCK_Q)
    s = scale.to(torch.float32).view(nb_out, 1, nb_in, 1)
    return (w * s).reshape(out_dim, in_dim)


def _swiglu_torch(g1):
    """Fallback SwiGLU for cuBLAS path."""
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

    # ── Stage 1: FP8 dequant → BF16 (halves bandwidth vs FP32) ──────────── #
    a = _dequant_hidden_states(hidden_states, hidden_states_scale)

    # ── Stage 2: Routing (fully vectorized) ──────────────────────────────── #
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

    # ── Stage 3: Build sorted expert layout ──────────────────────────────── #
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

    # ── Stage 4: Bulk gather (pays off when N_valid >= 64) ───────────────── #
    N_valid = sorted_token_idx.numel()
    use_bulk = (N_valid >= 64)

    if use_bulk:
        sorted_a = a.index_select(0, sorted_token_idx)
        sorted_w = topk_w[sorted_token_idx, sorted_topk_pos].to(torch.float32)

    # Pre-allocate scratch buffers (reused across all experts)
    max_tk = int(counts.max().item())
    c_buf = torch.empty((max_tk, INTERMEDIATE_SIZE), device=device, dtype=torch.float32)
    o_buf = torch.empty((max_tk, HIDDEN_SIZE), device=device, dtype=torch.float32)

    # ── Stage 5: Expert compute loop ─────────────────────────────────────── #
    start = 0
    for i in range(unique_experts.numel()):
        le = unique_experts[i].item()
        end = boundaries[i].item()
        Tk = end - start

        # Get this expert's tokens
        if use_bulk:
            a_e = sorted_a[start:end]        # contiguous slice, BF16
            w_e = sorted_w[start:end]
        else:
            token_idx_e = sorted_token_idx[start:end]
            a_e = a.index_select(0, token_idx_e)
            w_e = topk_w[token_idx_e, sorted_topk_pos[start:end]].to(torch.float32)

        t_idx = sorted_token_idx[start:end]

        if Tk >= FUSED_GEMM_THRESHOLD:
            # ── Fused Triton: GEMM1+SwiGLU (single kernel) ──
            c_view = c_buf[:Tk]
            _launch_gemm1_swiglu(
                a_e, gemm1_weights[le], gemm1_weights_scale[le],
                Tk, c_view
            )

            # ── Triton GEMM2 ──
            o_view = o_buf[:Tk]
            _launch_gemm2(
                c_view, gemm2_weights[le], gemm2_weights_scale[le],
                Tk, o_view
            )
        else:
            # ── cuBLAS fallback for tiny experts ──
            # Convert A to float32 for matmul precision
            a_e_f32 = a_e.to(torch.float32)

            w13_e = _dequant_weight(gemm1_weights[le], gemm1_weights_scale[le],
                                    2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
            g1 = torch.matmul(a_e_f32, w13_e.t())

            c_result = _swiglu_torch(g1)

            w2_e = _dequant_weight(gemm2_weights[le], gemm2_weights_scale[le],
                                   HIDDEN_SIZE, INTERMEDIATE_SIZE)
            o_view = torch.matmul(c_result, w2_e.t())

        # ── Weighted scatter-add ──
        accum.index_add_(0, t_idx, o_view * w_e.unsqueeze(1))

        start = end

    # ── Write BF16 ─────────────────────────────────────────────────────────── #
    output.copy_(accum.to(torch.bfloat16))