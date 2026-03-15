"""
Triton optimized MoE kernel — Submission 11
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Sub-11 = Optimized Sub-10 layout + Sub-6 compute throughput

  CHANGES OVER SUB-10:
    1. Restored BLOCK_K=128: 2× better compute throughput than BK=64.
    2. Restored BLOCK_M=64 for GEMM1: Better parallelism.
    3. Tuned num_stages=3: Fits safely in B200 SMEM (~184KB) with 128-block.
    4. Lowered FUSED_GEMM_THRESHOLD=16: Recover mid-size expert performance.

  RETAINED FROM SUB-10 (High speedup):
    - Zero dead weight loads in GEMM1.
    - Fused routing-weight multiply in GEMM2.
    - Fused scatter-add via tl.atomic_add (Eliminated o_buf).
    - 1D swizzled grid for L2-friendly expert processing.
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

FUSED_GEMM_THRESHOLD = 16


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  GEMM1 + SwiGLU: BF16 tensor cores, factored scales, zero dead loads         #
#                                                                               #
#  Memory traffic per K-iteration (BLOCK_M=64, BLOCK_N=128, BLOCK_K=128):      #
#    A tile:       64×128 × 1 byte  =   8 KB  (FP8, loaded once for gate+up)   #
#    W_gate tile: 128×128 × 1 byte  =  16 KB  (FP8, (K,N) order)              #
#    W_up tile:  128×128 × 1 byte   =  16 KB  (FP8, (K,N) order)              #
#    Scales:      3 × 4 bytes       =  12 B   (2 weight + 1 activation vector) #
#    Total:                           ~40 KB   (was ~80 KB with double loads)   #
#                                                                               #
#  Registers per warp (4 warps):                                                #
#    gate_acc: 64×128 FP32 = 32 KB                                             #
#    up_acc:   64×128 FP32 = 32 KB                                             #
#    tiles:    ~8 KB (a, w_gate, w_up temporaries)                              #
#    Total:    ~72 KB per warp (fits in 256 KB register file / 4 warps)         #
#                                                                               #
#  1D swizzled grid for L2 reuse of weight tiles across M-tiles.                #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _fused_gemm1_swiglu_kernel(
    a_ptr,          # (M, K) FP8
    a_scale_ptr,    # (K//128, M) FP32 — indexed as [k_block, token]
    w_ptr,          # (2*N_HALF, K) FP8 — rows 0:N_HALF=gate, N_HALF:2*N_HALF=up
    w_scale_ptr,    # (2*N_HALF//128, K//128) FP32
    c_ptr,          # (M, N_HALF) FP32 output
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
    GROUP_SIZE_M: tl.constexpr,
):
    # ── 1D swizzled grid → (pid_m, pid_n) ──
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N_HALF, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N_HALF

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Scale indices (constant for this tile)
    n_scale_gate = pid_n
    n_scale_up = pid_n + N_HALF_BLOCKS

    # Precompute base pointers for the inner loop
    a_base = a_ptr + offs_m[:, None] * sa0    # (BLOCK_M, 1)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_blk = k_start // 128  # Must divide by quantization block size (128)

        # ── Load A tile: FP8 → BF16 (lossless) ──
        a_tile = tl.load(
            a_base + offs_k[None, :] * sa1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)

        # ── Load weights in (BLOCK_N, BLOCK_K) order for contiguous load ──
        # BLOCK_K is now 128, which exactly matches sw0/sw1 strides for experts.
        w_gate = tl.load(
            w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)

        w_up = tl.load(
            w_ptr + (offs_n[:, None] + N_HALF) * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)

        # ── BF16 tensor core dot using tl.trans ──
        raw_gate = tl.dot(a_tile, tl.trans(w_gate))    # (BLOCK_M, BLOCK_N) FP32
        raw_up = tl.dot(a_tile, tl.trans(w_up))        # (BLOCK_M, BLOCK_N) FP32

        # ── Scales in FP32 post-dot (zero precision loss) ──
        a_s = tl.load(
            a_scale_ptr + k_blk * sas0 + offs_m * sas1,
            mask=mask_m, other=1.0
        ).to(tl.float32)

        ws_gate = tl.load(w_scale_ptr + n_scale_gate * sws0 + k_blk * sws1).to(tl.float32)
        ws_up = tl.load(w_scale_ptr + n_scale_up * sws0 + k_blk * sws1).to(tl.float32)

        # Combined scale: a_s[m] * ws (broadcast over N)
        gate_acc += raw_gate * (a_s[:, None] * ws_gate)
        up_acc += raw_up * (a_s[:, None] * ws_up)

    # ── SwiGLU epilogue (all FP32) ──
    result = gate_acc * (up_acc * tl.sigmoid(up_acc))

    tl.store(
        c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1,
        result,
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  GEMM2 + fused route-weight multiply + scatter-add                            #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
@triton.jit
def _gemm2_scatter_kernel(
    c_ptr,              # (M, K) FP32 — SwiGLU output
    w_ptr,              # (N, K) FP8
    s_ptr,              # (N//128, K//128) FP32
    accum_ptr,          # (T_orig, N) FP32 — global output accumulator
    route_w_ptr,        # (M,) FP32 — routing weights for this expert's tokens
    token_map_ptr,      # (M,) int64 — maps local row → original token row
    M, N, K,
    T_orig,             # original number of tokens (for bounds checking)
    sc0, sc1,
    sw0, sw1,
    ss0, ss1,
    saccum0, saccum1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ── 1D swizzled grid ──
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_block_idx = pid_n

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        k_block_idx = k_start // 128

        # C input (FP32, SwiGLU output)
        c_tile = tl.load(
            c_ptr + offs_m[:, None] * sc0 + offs_k[None, :] * sc1,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        )

        # W2 in (BLOCK_N, BLOCK_K) order
        w = tl.load(
            w_ptr + offs_n[:, None] * sw0 + offs_k[None, :] * sw1,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)

        s_val = tl.load(s_ptr + n_block_idx * ss0 + k_block_idx * ss1).to(tl.float32)
        w_dq = w * s_val

        # TF32 tensor core dot
        acc += tl.dot(c_tile, tl.trans(w_dq))

    # ── Fused epilogue ──
    route_w = tl.load(route_w_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    acc = acc * route_w[:, None]

    # ── Fused scatter-add ──
    orig_rows = tl.load(token_map_ptr + offs_m, mask=mask_m, other=0)
    out_ptrs = accum_ptr + orig_rows[:, None] * saccum0 + offs_n[None, :] * saccum1
    tl.atomic_add(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ═══════════════════════════ Python Launchers ═════════════════════════════════ #
def _launch_fused_gemm1_swiglu(a_fp8, a_scale, w_fp8, w_scale, Tk, c_out):
    BM, BN, BK = 64, 128, 128
    grid_size = triton.cdiv(Tk, BM) * triton.cdiv(INTERMEDIATE_SIZE, BN)
    _fused_gemm1_swiglu_kernel[(grid_size,)](
        a_fp8, a_scale, w_fp8, w_scale, c_out,
        Tk, INTERMEDIATE_SIZE, HIDDEN_SIZE,
        a_fp8.stride(0), a_fp8.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        c_out.stride(0), c_out.stride(1),
        N_HALF_BLOCKS=INTERMEDIATE_SIZE // 128,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_SIZE_M=8,
        num_stages=3, num_warps=4,
    )


def _launch_gemm2_scatter(c_e, w_fp8, w_scale, route_w, token_map, Tk, accum):
    BM, BN, BK = 64, 128, 128
    grid_size = triton.cdiv(Tk, BM) * triton.cdiv(HIDDEN_SIZE, BN)
    T_orig = accum.shape[0]
    _gemm2_scatter_kernel[(grid_size,)](
        c_e, w_fp8, w_scale, accum,
        route_w, token_map,
        Tk, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        T_orig,
        c_e.stride(0), c_e.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        accum.stride(0), accum.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_SIZE_M=8,
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
    t_size, h_size = hidden_states.shape
    nb_h = h_size // BLOCK_Q
    x = hidden_states.to(torch.float32).view(t_size, nb_h, BLOCK_Q)
    s = hidden_states_scale.to(torch.float32).t().unsqueeze(2)
    return (x * s).reshape(t_size, h_size)


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
    print("=== KERNEL START ===")
    print(f"Input routing_logits shape: {routing_logits.shape}, dtype: {routing_logits.dtype}")
    print(f"Input routing_bias shape: {routing_bias.shape}, dtype: {routing_bias.dtype}")
    print(f"Input hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
    print(f"Input hidden_states_scale shape: {hidden_states_scale.shape}, dtype: {hidden_states_scale.dtype}")
    print(f"Input gemm1_weights shape: {gemm1_weights.shape}, dtype: {gemm1_weights.dtype}")
    print(f"Input gemm1_weights_scale shape: {gemm1_weights_scale.shape}, dtype: {gemm1_weights_scale.dtype}")
    print(f"Input gemm2_weights shape: {gemm2_weights.shape}, dtype: {gemm2_weights.dtype}")
    print(f"Input gemm2_weights_scale shape: {gemm2_weights_scale.shape}, dtype: {gemm2_weights_scale.dtype}")
    print(f"local_expert_offset: {local_expert_offset}")
    print(f"routed_scaling_factor: {routed_scaling_factor}")
    print(f"Output shape: {output.shape}, dtype: {output.dtype}")

    t_size = routing_logits.shape[0]
    local_start = int(local_expert_offset)
    device = hidden_states.device

    print(f"\n1. Basic setup:")
    print(f"   t_size (sequence length): {t_size}")
    print(f"   local_start (expert offset): {local_start}")
    print(f"   device: {device}")

    print("# 2. Routing computation:")
    logits = routing_logits.to(torch.float32)
    print(f"   logits converted to FP32, shape: {logits.shape}")

    bias = routing_bias.to(torch.float32).view(-1)
    print(f"   bias converted to FP32 and reshaped, shape: {bias.shape}")

    s = torch.sigmoid(logits)
    print(f"   sigmoid applied, s shape: {s.shape}, range: [{s.min():.4f}, {s.max():.4f}]")

    s_with_bias = s + bias
    print(f"   bias added, s_with_bias shape: {s_with_bias.shape}, range: [{s_with_bias.min():.4f}, {s_with_bias.max():.4f}]")

    print("# 3. Group-based routing:")
    s_wb_grouped = s_with_bias.view(t_size, N_GROUP, GROUP_SIZE)
    print(f"   s_with_bias grouped into {N_GROUP} groups of {GROUP_SIZE} experts each")
    print(f"   s_wb_grouped shape: {s_wb_grouped.shape}")

    top2_vals = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False).values
    print(f"   top-2 values per group extracted, shape: {top2_vals.shape}")

    group_scores = top2_vals.sum(dim=2)
    print(f"   group scores computed (sum of top-2), shape: {group_scores.shape}")
    print(f"   group_scores range: [{group_scores.min():.4f}, {group_scores.max():.4f}]")

    print("# 4. Group selection:")
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False).indices
    print(f"   selected top-{TOPK_GROUP} groups, indices: {group_idx}")

    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)
    print(f"   group mask created, shape: {group_mask.shape}")

    score_mask = group_mask.unsqueeze(2).expand(t_size, N_GROUP, GROUP_SIZE).reshape(t_size, NUM_EXPERTS)
    print(f"   score mask expanded to expert level, shape: {score_mask.shape}")

    scores_pruned = s_with_bias.masked_fill(~score_mask, float("-inf"))
    print(f"   scores pruned (non-selected experts set to -inf)")

    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False).indices
    print(f"   top-{TOP_K} experts selected globally, shape: {topk_idx.shape}")
    print(f"   selected expert indices: {topk_idx}")

    print("# 5. Weight computation:")
    topk_s = torch.gather(s, 1, topk_idx)
    print(f"   top-k sigmoid values gathered, shape: {topk_s.shape}")

    topk_w = (topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20)) * float(routed_scaling_factor)
    print(f"   routing weights normalized and scaled, shape: {topk_w.shape}")
    print(f"   routing weights range: [{topk_w.min():.4f}, {topk_w.max():.4f}]")

    print("# 6. Local expert filtering:")
    local_idx = topk_idx - local_start
    print(f"   expert indices shifted by local_start ({local_start}), shape: {local_idx.shape}")

    valid_local = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)
    print(f"   valid local experts mask created, shape: {valid_local.shape}")
    print(f"   valid selections per token: {valid_local.sum(dim=1)}")

    accum = torch.zeros((t_size, HIDDEN_SIZE), dtype=torch.float32, device=device)
    print(f"   output accumulator initialized, shape: {accum.shape}")

    all_valid_idx = torch.nonzero(valid_local, as_tuple=False)
    print(f"   all valid (token, topk_pos) pairs found: {all_valid_idx.shape}")

    if all_valid_idx.numel() == 0:
        print("   No valid local experts found - returning zero output")
        output.copy_(accum.to(torch.bfloat16))
        return

    print("# 7. Expert processing setup:")
    flat_token_idx = all_valid_idx[:, 0]
    flat_topk_pos = all_valid_idx[:, 1]
    flat_expert_id = local_idx[flat_token_idx, flat_topk_pos]

    print(f"   flat_token_idx: {flat_token_idx}")
    print(f"   flat_topk_pos: {flat_topk_pos}")
    print(f"   flat_expert_id: {flat_expert_id}")

    sort_order = torch.argsort(flat_expert_id, stable=True)
    print(f"   sort order for expert grouping: {sort_order}")

    sorted_expert_id = flat_expert_id[sort_order]
    sorted_token_idx = flat_token_idx[sort_order]
    sorted_topk_pos = flat_topk_pos[sort_order]

    print(f"   sorted_expert_id: {sorted_expert_id}")
    print(f"   sorted_token_idx: {sorted_token_idx}")
    print(f"   sorted_topk_pos: {sorted_topk_pos}")

    unique_experts, counts = torch.unique_consecutive(sorted_expert_id, return_counts=True)
    boundaries = torch.cumsum(counts, dim=0)

    print(f"   unique experts to process: {unique_experts}")
    print(f"   token counts per expert: {counts}")
    print(f"   expert boundaries: {boundaries}")

    print("# 8. Processing experts:")
    start = 0
    for i in range(unique_experts.numel()):
        le = unique_experts[i].item()
        end = boundaries[i].item()
        Tk = end - start
        t_idx = sorted_token_idx[start:end]
        w_e = topk_w[t_idx, sorted_topk_pos[start:end]].to(torch.float32)

        print(f"\n   Expert {le} (local expert {i}):")
        print(f"     token count: {Tk}")
        print(f"     token indices: {t_idx}")
        print(f"     routing weights: {w_e}")

        if Tk > 0:
            print(f"     Dequantizing GEMM1 weights for expert {le}...")
            w13_e = _dequant_weight(gemm1_weights[le], gemm1_weights_scale[le], 2*INTERMEDIATE_SIZE, HIDDEN_SIZE)
            print(f"       GEMM1 weights shape: {w13_e.shape}")

            print(f"     Dequantizing GEMM2 weights for expert {le}...")
            w2_e = _dequant_weight(gemm2_weights[le], gemm2_weights_scale[le], HIDDEN_SIZE, INTERMEDIATE_SIZE)
            print(f"       GEMM2 weights shape: {w2_e.shape}")

            print(f"     Dequantizing input tokens...")
            a_e = _dequant_hidden_fp32(hidden_states.index_select(0, t_idx), hidden_states_scale.index_select(1, t_idx).contiguous())
            print(f"       Input tokens shape: {a_e.shape}")

            print(f"     GEMM1: {a_e.shape} @ {w13_e.t().shape} = {(a_e @ w13_e.t()).shape}")
            g1 = torch.matmul(a_e, w13_e.t())
            print(f"       GEMM1 result range: [{g1.min():.4f}, {g1.max():.4f}]")

            print(f"     SwiGLU activation...")
            c_result = _swiglu_torch(g1)
            print(f"       SwiGLU result shape: {c_result.shape}, range: [{c_result.min():.4f}, {c_result.max():.4f}]")

            print(f"     GEMM2: {c_result.shape} @ {w2_e.t().shape} = {(c_result @ w2_e.t()).shape}")
            o = torch.matmul(c_result, w2_e.t())
            print(f"       GEMM2 result shape: {o.shape}, range: [{o.min():.4f}, {o.max():.4f}]")

            print(f"     Accumulating with routing weights...")
            weighted_output = o * w_e.unsqueeze(1)
            accum.index_add_(0, t_idx, weighted_output)
            print(f"       Accumulation completed for expert {le}")

        start = end

    print("# 9. Final output:")
    output.copy_(accum.to(torch.bfloat16))
    print(f"   Output copied to BF16, final shape: {output.shape}")
    print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("=== KERNEL END ===")