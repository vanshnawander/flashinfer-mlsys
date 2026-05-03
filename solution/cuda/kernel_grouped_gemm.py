"""
kernel_grouped_gemm.py — Optimized MoE FP8 Grouped GEMM kernel wrapper
Target: NVIDIA B200 (SM 10.0 / sm_100a)

Entry point: kernel() — DPS-style, 11 parameters matching the harness.

Build flow:
  1. Try to JIT-compile kernel_grouped_gemm.cu with B200 flags (-arch=sm_100a).
  2. If the GPU is Hopper (SM 9.0), fall back to -arch=sm_90a.
  3. If compilation fails entirely, use the pure-PyTorch fallback below.

Architecture (CUDA path):
  - Fused sigmoid+bias kernel
  - FP8→FP32 dequant via __nv_fp8_e4m3 intrinsics (cuda_fp8.h, CUDA 11.8+)
  - Grouped GEMM1: batched mm over all experts (pre-dequanted weights)
  - Batched SwiGLU: single kernel over all tokens
  - Grouped GEMM2: batched mm
  - Batched weighted scatter-add: single kernel
"""

import torch
import torch.nn.functional as F
import os

# ═══════════════════════════ Constants ══════════════════════════════════════ #
HIDDEN_SIZE       = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS       = 256
NUM_LOCAL_EXPERTS = 32
BLOCK_Q           = 128
TOP_K             = 8
N_GROUP           = 8
TOPK_GROUP        = 4
GROUP_SIZE        = NUM_EXPERTS // N_GROUP


# ═══════════════════════════ Extension Loading ═══════════════════════════════ #
_cuda_ext = None
_build_attempted = False


def _detect_sm() -> str:
    """Detect SM version for the current CUDA device."""
    if not torch.cuda.is_available():
        return "sm_90a"
    major, minor = torch.cuda.get_device_capability()
    if major == 10:           # Blackwell (B200)
        return "sm_100a"
    elif major == 9:          # Hopper (H100)
        return "sm_90a"
    elif major == 8 and minor >= 9:  # Ada (L40 etc.)
        return "sm_89"
    else:
        return f"sm_{major}{minor}"


def _load_cuda_extension():
    """Lazy JIT-compile kernel_grouped_gemm.cu. Cached after first call."""
    global _cuda_ext, _build_attempted
    if _build_attempted:
        return _cuda_ext
    _build_attempted = True

    try:
        from torch.utils.cpp_extension import load as _load_ext
        _cuda_dir = os.path.dirname(os.path.abspath(__file__))
        cu_src    = os.path.join(_cuda_dir, "kernel_grouped_gemm.cu")

        arch = _detect_sm()
        print(f"[MoE CUDA] Compiling kernel_grouped_gemm.cu for {arch} ...")

        _cuda_ext = _load_ext(
            name="moe_grouped_gemm_v2",
            sources=[cu_src],
            verbose=True,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                f"-arch={arch}",
                "-std=c++17",
                # Allow __nv_fp8_e4m3 on SM < 89 as software path
                "-DCUDA_NO_HALF",       # prevent stale half-precision macros
            ],
            extra_cflags=["-O3", "-std=c++17"],
        )
        print(f"[MoE CUDA] Compiled OK ({arch})")
        return _cuda_ext
    except Exception as exc:
        print(f"[MoE CUDA] Compilation failed: {exc}")
        print("[MoE CUDA] Falling back to pure-PyTorch path")
        return None


# ═══════════════════════ Pure-PyTorch Fallback Helpers ═══════════════════════ #

def _dequant_weight(w_fp8, scale, out_dim: int, in_dim: int) -> torch.Tensor:
    """FP8 block-scale weight dequant → FP32."""
    nb_out = out_dim // BLOCK_Q
    nb_in  = in_dim  // BLOCK_Q
    w = w_fp8.to(torch.float32).view(nb_out, BLOCK_Q, nb_in, BLOCK_Q)
    s = scale.to(torch.float32).view(nb_out, 1, nb_in, 1)
    return (w * s).reshape(out_dim, in_dim)


def _dequant_hidden(hidden_states: torch.Tensor,
                    hidden_states_scale: torch.Tensor) -> torch.Tensor:
    """FP8 block-scale hidden states dequant → FP32."""
    T, H = hidden_states.shape
    nb_h = H // BLOCK_Q
    x = hidden_states.to(torch.float32).view(T, nb_h, BLOCK_Q)
    s = hidden_states_scale.to(torch.float32).t().unsqueeze(2)
    return (x * s).reshape(T, H)


def _swiglu(g1: torch.Tensor) -> torch.Tensor:
    """SwiGLU: gate * silu(up).  g1 shape: (M, 2*I)."""
    I    = g1.shape[1] // 2
    gate = g1[:, :I]
    up   = g1[:,  I:]
    return (gate * F.silu(up)).to(torch.float32)


# ═══════════════════════════════  MAIN KERNEL ═══════════════════════════════ #

@torch.no_grad()
def kernel(
    routing_logits:        torch.Tensor,   # (T, 256) float32
    routing_bias:          torch.Tensor,   # (256,) float32 or bfloat16
    hidden_states:         torch.Tensor,   # (T, H)   FP8 E4M3
    hidden_states_scale:   torch.Tensor,   # (H//128, T) float32
    gemm1_weights:         torch.Tensor,   # (32, 2I, H)  FP8 E4M3
    gemm1_weights_scale:   torch.Tensor,   # (32, 2I//128, H//128) float32
    gemm2_weights:         torch.Tensor,   # (32, H, I)   FP8 E4M3
    gemm2_weights_scale:   torch.Tensor,   # (32, H//128, I//128) float32
    local_expert_offset:   int,
    routed_scaling_factor: float,
    output:                torch.Tensor,   # (T, H) BF16  — DPS pre-allocated
):
    """
    MoE FP8 kernel — try CUDA extension, fall back to grouped-PyTorch.

    The CUDA extension (kernel_grouped_gemm.cu) handles:
      - Routing with fused sigmoid+bias kernel
      - FP8 dequant via __nv_fp8_e4m3 (no manual bit-twiddling)
      - Grouped GEMM (batched mm, pre-allocated contiguous buffers)
      - Single SwiGLU + scatter-add kernel launch
    """

    # ── Try CUDA extension ───────────────────────────────────────────────── #
    ext = _load_cuda_extension()
    if ext is not None:
        try:
            ext.kernel(
                routing_logits,
                routing_bias,
                hidden_states.contiguous(),
                hidden_states_scale.contiguous(),
                gemm1_weights,
                gemm1_weights_scale,
                gemm2_weights,
                gemm2_weights_scale,
                int(local_expert_offset),
                float(routed_scaling_factor),
                output,
            )
            return
        except Exception as exc:
            print(f"[MoE CUDA] kernel() raised: {exc} — using PyTorch fallback")

    # ── Pure-PyTorch grouped GEMM path ──────────────────────────────────── #
    t_size      = routing_logits.shape[0]
    local_start = int(local_expert_offset)
    device      = hidden_states.device

    hidden_states       = hidden_states.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()

    torch.backends.cuda.matmul.allow_tf32 = False

    # Stage 1: Routing
    logits      = routing_logits.to(torch.float32)
    bias        = routing_bias.to(torch.float32).view(-1)
    s           = torch.sigmoid(logits)
    s_with_bias = s + bias

    s_wb_grouped = s_with_bias.view(t_size, N_GROUP, GROUP_SIZE)
    top2_vals    = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2_vals.sum(dim=2)

    group_idx   = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False).indices
    group_mask  = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)

    score_mask    = group_mask.unsqueeze(2).expand(t_size, N_GROUP, GROUP_SIZE).reshape(t_size, NUM_EXPERTS)
    scores_pruned = s_with_bias.masked_fill(~score_mask, float("-inf"))
    topk_idx      = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False).indices

    topk_s = torch.gather(s, 1, topk_idx)
    topk_w = topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20)
    topk_w = topk_w * float(routed_scaling_factor)

    # Stage 2: Dispatch table
    local_idx   = topk_idx - local_start
    valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)
    accum = torch.zeros((t_size, HIDDEN_SIZE), dtype=torch.float32, device=device)

    all_valid_idx = torch.nonzero(valid_local, as_tuple=False)
    if all_valid_idx.numel() == 0:
        output.copy_(accum.to(torch.bfloat16))
        return

    flat_token_idx   = all_valid_idx[:, 0]
    flat_topk_pos    = all_valid_idx[:, 1]
    flat_expert_id   = local_idx[flat_token_idx, flat_topk_pos]

    sort_order       = torch.argsort(flat_expert_id, stable=True)
    sorted_expert_id = flat_expert_id[sort_order]
    sorted_token_idx = flat_token_idx[sort_order]
    sorted_topk_pos  = flat_topk_pos[sort_order]

    unique_experts, counts = torch.unique_consecutive(sorted_expert_id, return_counts=True)
    boundaries  = torch.cumsum(counts, dim=0)
    N_valid     = sorted_token_idx.numel()
    num_unique   = unique_experts.numel()

    counts_cpu         = counts.cpu()
    boundaries_cpu     = boundaries.cpu()
    unique_experts_cpu = unique_experts.cpu()

    # Stage 3: Bulk FP8 dequant + gather
    a_fp32   = _dequant_hidden(hidden_states, hidden_states_scale)
    sorted_a = a_fp32.index_select(0, sorted_token_idx)
    sorted_w = topk_w[sorted_token_idx, sorted_topk_pos].to(torch.float32)

    # Pre-dequant all expert weights
    w13_dequant = [
        _dequant_weight(gemm1_weights[unique_experts_cpu[i].item()],
                        gemm1_weights_scale[unique_experts_cpu[i].item()],
                        2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
        for i in range(num_unique)
    ]
    w2_dequant = [
        _dequant_weight(gemm2_weights[unique_experts_cpu[i].item()],
                        gemm2_weights_scale[unique_experts_cpu[i].item()],
                        HIDDEN_SIZE, INTERMEDIATE_SIZE)
        for i in range(num_unique)
    ]

    # Stage 4: Grouped GEMM1
    g1_all = torch.empty((N_valid, 2 * INTERMEDIATE_SIZE), dtype=torch.float32, device=device)
    start = 0
    for i in range(num_unique):
        end = boundaries_cpu[i].item()
        torch.mm(sorted_a[start:end], w13_dequant[i].t(), out=g1_all[start:end])
        start = end

    # Stage 5: Batched SwiGLU
    c_all = _swiglu(g1_all)

    # Stage 6: Grouped GEMM2
    o_all = torch.empty((N_valid, HIDDEN_SIZE), dtype=torch.float32, device=device)
    start = 0
    for i in range(num_unique):
        end = boundaries_cpu[i].item()
        torch.mm(c_all[start:end], w2_dequant[i].t(), out=o_all[start:end])
        start = end

    # Stage 7: Weighted scatter-add
    accum.index_add_(0, sorted_token_idx, o_all * sorted_w.unsqueeze(1))

    output.copy_(accum.to(torch.bfloat16))
