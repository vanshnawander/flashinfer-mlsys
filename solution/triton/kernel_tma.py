"""
Triton B200 TMA-Optimized MoE Kernel
Based on Sub-9 Math (Lossless FP8->BF16) + TMA Block Pointers + Safe Epilogue Fusion
"""

import torch
import triton
import triton.language as tl

# Constants
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
#  FUSED GEMM1 + SwiGLU with TMA Block Pointers                             #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _fused_gemm1_swiglu_tma_kernel(
    a_ptr, a_scale_ptr, w_ptr, w_scale_ptr, c_ptr,
    M, N_HALF, K,
    sa0, sa1, sas0, sas1, sw0, sw1, sws0, sws1, sc0, sc1,
    N_HALF_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # ── TMA Block Pointers ──
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(sa0, sa1),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    
    # We define W block pointers as Transposed (K, N) to feed tl.dot cleanly
    w_gate_ptr = tl.make_block_ptr(
        base=w_ptr, shape=(K, N_HALF), strides=(sw1, sw0),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1)
    )
    
    w_up_ptr = tl.make_block_ptr(
        base=w_ptr + N_HALF * sw0, shape=(K, N_HALF), strides=(sw1, sw0),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1)
    )

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_scale_gate = pid_n
    n_scale_up = pid_n + N_HALF_BLOCKS

    for k_start in range(0, K, BLOCK_K):
        k_blk = k_start // BLOCK_K

        # Hardware TMA asynchronous loads
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1)).to(tl.bfloat16)
        w_gate_t = tl.load(w_gate_ptr, boundary_check=(0, 1)).to(tl.bfloat16)
        w_up_t = tl.load(w_up_ptr, boundary_check=(0, 1)).to(tl.bfloat16)

        # BF16 Tensor Cores (WGMMA generated automatically)
        raw_gate = tl.dot(a_tile, w_gate_t)
        raw_up = tl.dot(a_tile, w_up_t)

        # Manual pointer loads for scales (too small for TMA)
        a_s = tl.load(a_scale_ptr + k_blk * sas0 + offs_m * sas1, mask=mask_m, other=1.0).to(tl.float32)
        ws_gate = tl.load(w_scale_ptr + n_scale_gate * sws0 + k_blk * sws1).to(tl.float32)
        ws_up = tl.load(w_scale_ptr + n_scale_up * sws0 + k_blk * sws1).to(tl.float32)

        gate_acc += raw_gate * (a_s[:, None] * ws_gate)
        up_acc += raw_up * (a_s[:, None] * ws_up)

        # Advance TMA pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        w_gate_ptr = tl.advance(w_gate_ptr, (BLOCK_K, 0))
        w_up_ptr = tl.advance(w_up_ptr, (BLOCK_K, 0))

    # FP32 SwiGLU
    result = gate_acc * (up_acc * tl.sigmoid(up_acc))

    # TMA Store
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N_HALF), strides=(sc0, sc1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    tl.store(c_block_ptr, result, boundary_check=(0, 1))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  GEMM2 + FUSED ROUTE WEIGHT with TMA Block Pointers                       #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _gemm2_weighted_tma_kernel(
    c_ptr, w_ptr, s_ptr, route_w_ptr, o_ptr,
    M, N, K,
    sc0, sc1, sw0, sw1, ss0, ss1, so0, so1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, K), strides=(sc0, sc1),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    
    # We don't use TMA for W here because we need to dequantize it to FP32 *before* 
    # dotting with the FP32 SwiGLU input C. Mixed precision dot is not allowed.
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_block_idx = pid_n

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_block_idx = k_start // 128

        c_tile = tl.load(c_block_ptr, boundary_check=(0, 1)) # FP32

        w_tile = tl.load(w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1, 
                         mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        s_tile = tl.load(s_ptr + n_block_idx * ss0 + k_block_idx * ss1, mask=mask_n, other=1.0).to(tl.float32)
        
        w_dequant = w_tile * s_tile[:, None]
        acc += tl.dot(c_tile, tl.trans(w_dequant)) # TF32 Tensor Cores
        
        c_block_ptr = tl.advance(c_block_ptr, (0, BLOCK_K))

    # ── Safe In-Register Epilogue Fusion ──
    route_w = tl.load(route_w_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    acc = acc * route_w[:, None]

    o_block_ptr = tl.make_block_ptr(
        base=o_ptr, shape=(M, N), strides=(so0, so1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    tl.store(o_block_ptr, acc, boundary_check=(0, 1))

# --- Launchers ---
def _launch_fused_gemm1_swiglu_tma(a_fp8, a_scale, w_fp8, w_scale, Tk, c_out):
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(Tk, BM), triton.cdiv(INTERMEDIATE_SIZE, BN))
    _fused_gemm1_swiglu_tma_kernel[grid](
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

def _launch_gemm2_weighted_tma(c_e, w_fp8, w_scale, route_w, Tk, o_out):
    BM, BN, BK = 64, 128, 128
    grid = (triton.cdiv(Tk, BM), triton.cdiv(HIDDEN_SIZE, BN))
    _gemm2_weighted_tma_kernel[grid](
        c_e, w_fp8, w_scale, route_w, o_out,
        Tk, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        c_e.stride(0), c_e.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        o_out.stride(0), o_out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_stages=3, num_warps=4, # Kept safe to avoid SMEM spills
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

    # ══════════════════════════════════════════════════════════════════════ #
    #  ROUTING                                                              #
    # ══════════════════════════════════════════════════════════════════════ #
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

    # ══════════════════════════════════════════════════════════════════════ #
    #  DISPATCH                                                             #
    # ══════════════════════════════════════════════════════════════════════ #
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

    # ══════════════════════════════════════════════════════════════════════ #
    #  BULK GATHER                                                          #
    # ══════════════════════════════════════════════════════════════════════ #
    N_valid = sorted_token_idx.numel()
    use_bulk = (N_valid >= 64)

    if use_bulk:
        sorted_a_fp8 = hidden_states.index_select(0, sorted_token_idx)
        sorted_a_scale = hidden_states_scale.index_select(1, sorted_token_idx)
        sorted_w = topk_w[sorted_token_idx, sorted_topk_pos].to(torch.float32)

    a_fp32_cache = None

    # Pre-allocate scratch — ALL FP32
    max_tk = int(counts.max().item())
    c_buf = torch.empty((max_tk, INTERMEDIATE_SIZE), device=device, dtype=torch.float32)
    o_buf = torch.empty((max_tk, HIDDEN_SIZE), device=device, dtype=torch.float32)

    # ══════════════════════════════════════════════════════════════════════ #
    #  PER-EXPERT COMPUTE                                                   #
    # ══════════════════════════════════════════════════════════════════════ #
    start = 0
    for i in range(unique_experts.numel()):
        le = unique_experts[i].item()
        end = boundaries[i].item()
        Tk = end - start

        t_idx = sorted_token_idx[start:end]

        if Tk >= FUSED_GEMM_THRESHOLD:
            if use_bulk:
                a_e_fp8 = sorted_a_fp8[start:end]
                a_e_scale = sorted_a_scale[:, start:end]
                w_e = sorted_w[start:end]
            else:
                a_e_fp8 = hidden_states.index_select(0, t_idx)
                a_e_scale = hidden_states_scale.index_select(1, t_idx)
                w_e = topk_w[t_idx, sorted_topk_pos[start:end]].to(torch.float32)

            # Fused GEMM1+SwiGLU → FP32
            c_view = c_buf[:Tk]
            _launch_fused_gemm1_swiglu_tma(
                a_e_fp8, a_e_scale,
                gemm1_weights[le], gemm1_weights_scale[le],
                Tk, c_view
            )

            # GEMM2 with fused route-weight → FP32
            o_view = o_buf[:Tk]
            _launch_gemm2_weighted_tma(
                c_view, gemm2_weights[le], gemm2_weights_scale[le],
                w_e, Tk, o_view
            )

            # Route-weight already applied in GEMM2 epilogue
            accum.index_add_(0, t_idx, o_view)

        else:
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