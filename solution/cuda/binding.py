"""
CUDA binding for fused MoE kernel — B200 Compatible
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Thin Python entry point: routing logic only in Python, all heavy
compute (dequant, GEMM, SwiGLU, scatter, cast) delegated to kernel.cu.

Architecture:
    binding.py (this file):
        - Routing logic (lightweight PyTorch ops — ~5% of total time)
        - Expert dispatch table construction
        - Orchestration of CUDA kernel calls

    kernel.cu (compiled extension):
        - dequant_hidden_states: FP8→FP32 vectorized dequant
        - dequant_weights: FP8→FP32 weight dequant
        - swiglu: Fused SwiGLU activation
        - weighted_scatter_add: Fused weight × scatter-add
        - cast_to_bf16: FP32→BF16 vectorized cast
        - fused_dequant_matmul: FP8 dequant + matmul (avoids FP32 materialization)

Fallback: If CUDA extension fails to compile, pure PyTorch ops are used.
"""

import torch
import torch.nn.functional as F
import os

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


# ═══════════════════════════ Extension Loading ═══════════════════════════════ #

_cuda_ext = None

def _load_cuda_extension():
    """Lazy load the CUDA extension. Compile on first use."""
    global _cuda_ext
    if _cuda_ext is not None:
        return _cuda_ext

    try:
        from torch.utils.cpp_extension import load as _load_ext
        _cuda_dir = os.path.dirname(os.path.abspath(__file__))
        _cuda_ext = _load_ext(
            name='moe_cuda_kernels',
            sources=[os.path.join(_cuda_dir, 'kernel.cu')],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
        return _cuda_ext
    except Exception as e:
        print(f"[MoE CUDA] Extension compile failed: {e}")
        print("[MoE CUDA] Falling back to PyTorch ops")
        return None


# ═══════════════════════ PyTorch Fallbacks ═══════════════════════════════════ #

def _dequant_hidden_torch(hidden_states, hidden_states_scale):
    """FP8→FP32 dequant fallback."""
    T, H = hidden_states.shape
    nb_h = H // BLOCK_Q
    x = hidden_states.to(torch.float32).view(T, nb_h, BLOCK_Q)
    s = hidden_states_scale.to(torch.float32).t().unsqueeze(2)
    return (x * s).reshape(T, H)


def _dequant_weight_torch(w_fp8, scale, out_dim, in_dim):
    """FP8→FP32 weight dequant fallback."""
    nb_out = out_dim // BLOCK_Q
    nb_in = in_dim // BLOCK_Q
    w = w_fp8.to(torch.float32).view(nb_out, BLOCK_Q, nb_in, BLOCK_Q)
    s = scale.to(torch.float32).view(nb_out, 1, nb_in, 1)
    return (w * s).reshape(out_dim, in_dim)


def _swiglu_torch(g1):
    """SwiGLU fallback."""
    x1 = g1[:, :INTERMEDIATE_SIZE]
    x2 = g1[:, INTERMEDIATE_SIZE:]
    return (x1 * F.silu(x2)).to(torch.float32)


# ═══════════════════════ Compute Functions ═══════════════════════════════════ #

def _dequant_hidden(hidden_states, hidden_states_scale):
    """Dequant hidden states — CUDA if available, else PyTorch."""
    ext = _load_cuda_extension()
    if ext is not None:
        T, H = hidden_states.shape
        return ext.dequant_hidden_states(
            hidden_states.view(torch.uint8),
            hidden_states_scale,
            T, H
        )
    return _dequant_hidden_torch(hidden_states, hidden_states_scale)


def _dequant_weight(w_fp8, scale, out_dim, in_dim):
    """Dequant weights — CUDA if available, else PyTorch."""
    ext = _load_cuda_extension()
    if ext is not None:
        return ext.dequant_weights(
            w_fp8.view(torch.uint8),
            scale,
            out_dim, in_dim
        )
    return _dequant_weight_torch(w_fp8, scale, out_dim, in_dim)


def _swiglu(g1):
    """SwiGLU — CUDA if available, else PyTorch."""
    ext = _load_cuda_extension()
    if ext is not None:
        return ext.swiglu(g1, INTERMEDIATE_SIZE)
    return _swiglu_torch(g1)


def _scatter_add(expert_output, weight, token_idx, accum, Tk, H):
    """Weighted scatter-add — CUDA if available, else PyTorch."""
    ext = _load_cuda_extension()
    if ext is not None:
        ext.weighted_scatter_add(expert_output, weight, token_idx, accum, Tk, H)
    else:
        accum.index_add_(0, token_idx, expert_output * weight.unsqueeze(1))


def _cast_bf16(accum, output):
    """FP32→BF16 cast — CUDA if available, else PyTorch."""
    ext = _load_cuda_extension()
    if ext is not None:
        ext.cast_to_bf16(accum, output)
    else:
        output.copy_(accum.to(torch.bfloat16))


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

    # ── Dequant hidden states → FP32 (CUDA kernel) ──
    hidden_states = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    a = _dequant_hidden(hidden_states, hidden_states_scale)

    # ══════════════════════════════════════════════════════════════════════ #
    #  ROUTING — lightweight PyTorch ops (stays in Python)                  #
    # ══════════════════════════════════════════════════════════════════════ #
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).view(-1)
    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    s_wb_grouped = s_with_bias.view(t_size, N_GROUP, GROUP_SIZE)
    top2_vals = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2_vals.sum(dim=2)

    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)

    score_mask = group_mask.unsqueeze(2).expand(t_size, N_GROUP, GROUP_SIZE).reshape(t_size, NUM_EXPERTS)
    scores_pruned = s_with_bias.masked_fill(~score_mask, float("-inf"))
    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False).indices

    topk_s = torch.gather(s, 1, topk_idx)
    topk_w = topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * float(routed_scaling_factor)

    # ══════════════════════════════════════════════════════════════════════ #
    #  DISPATCH — build sorted expert layout                                 #
    # ══════════════════════════════════════════════════════════════════════ #
    local_idx = topk_idx - local_start
    valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)
    accum = torch.zeros((t_size, HIDDEN_SIZE), dtype=torch.float32, device=device)

    all_valid_idx = torch.nonzero(valid_local, as_tuple=False)
    if all_valid_idx.numel() == 0:
        _cast_bf16(accum, output)
        return

    flat_token_idx = all_valid_idx[:, 0]
    flat_topk_pos = all_valid_idx[:, 1]
    flat_expert_id = local_idx[flat_token_idx, flat_topk_pos]

    sort_order = torch.argsort(flat_expert_id, stable=True)
    sorted_expert_id = flat_expert_id[sort_order]
    sorted_token_idx = flat_token_idx[sort_order]
    sorted_topk_pos = flat_topk_pos[sort_order]

    # Bulk gather (FP32 — already dequanted)
    sorted_a = a.index_select(0, sorted_token_idx)
    sorted_w = topk_w[sorted_token_idx, sorted_topk_pos].to(torch.float32)

    unique_experts, counts = torch.unique_consecutive(sorted_expert_id, return_counts=True)
    boundaries = torch.cumsum(counts, dim=0)

    # Pre-allocate scratch buffers (reused across experts)
    max_tk = int(counts.max().item())
    g1_buf = torch.empty((max_tk, 2 * INTERMEDIATE_SIZE), device=device, dtype=torch.float32)
    o_buf = torch.empty((max_tk, HIDDEN_SIZE), device=device, dtype=torch.float32)

    # ══════════════════════════════════════════════════════════════════════ #
    #  PER-EXPERT COMPUTE                                                    #
    #  All heavy ops delegated to CUDA kernels:                              #
    #    1. dequant_weights → CUDA kernel                                    #
    #    2. matmul → cuBLAS (via torch.matmul)                              #
    #    3. SwiGLU → CUDA kernel                                             #
    #    4. scatter-add → CUDA kernel                                        #
    # ══════════════════════════════════════════════════════════════════════ #
    start = 0
    for i in range(unique_experts.numel()):
        le = unique_experts[i].item()
        end = boundaries[i].item()
        Tk = end - start
        a_e = sorted_a[start:end]
        t_idx = sorted_token_idx[start:end]
        w_e = sorted_w[start:end]

        # GEMM1: dequant weights (CUDA) + cuBLAS matmul
        w13_e = _dequant_weight(
            gemm1_weights[le], gemm1_weights_scale[le],
            2 * INTERMEDIATE_SIZE, HIDDEN_SIZE
        )
        g1_view = g1_buf[:Tk]
        torch.matmul(a_e, w13_e.t(), out=g1_view)

        # SwiGLU (CUDA kernel)
        c_result = _swiglu(g1_view)

        # GEMM2: dequant weights (CUDA) + cuBLAS matmul
        w2_e = _dequant_weight(
            gemm2_weights[le], gemm2_weights_scale[le],
            HIDDEN_SIZE, INTERMEDIATE_SIZE
        )
        o_view = o_buf[:Tk]
        torch.matmul(c_result, w2_e.t(), out=o_view)

        # Weighted scatter-add (CUDA kernel)
        _scatter_add(o_view, w_e, t_idx, accum, Tk, HIDDEN_SIZE)

        start = end

    # Final cast (CUDA kernel)
    _cast_bf16(accum, output)
