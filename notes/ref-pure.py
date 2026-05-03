"""
Pure NumPy/Python translation of the MoE kernel.
Every operation is explicit — no Triton, no CUDA, no magic.

This implements one MoE FFN layer from DeepSeek-V3/R1:
  - 256 total experts, 32 local to this GPU
  - Top-8 expert selection with grouped routing
  - FP8 block-scaled weights and activations
  - SwiGLU activation function
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════ #
#  CONSTANTS — these define the model architecture                           #
# ═══════════════════════════════════════════════════════════════════════════ #
HIDDEN_SIZE = 7168          # Dimension of each token's hidden state
INTERMEDIATE_SIZE = 2048    # Dimension inside each expert's FFN
NUM_EXPERTS = 256           # Total experts across all GPUs
NUM_LOCAL_EXPERTS = 32      # Experts on THIS GPU
BLOCK_Q = 128               # FP8 block-scale quantum: 1 scale per 128 values
TOP_K = 8                   # Each token picks 8 experts
N_GROUP = 8                 # Experts are divided into 8 groups
TOPK_GROUP = 4              # Each token picks from 4 of the 8 groups
GROUP_SIZE = NUM_EXPERTS // N_GROUP  # = 32 experts per group


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * sigmoid(x)


# ═══════════════════════════════════════════════════════════════════════════ #
#  STEP 0: Understanding the Inputs                                          #
# ═══════════════════════════════════════════════════════════════════════════ #
def explain_inputs():
    """
    This function doesn't compute anything — it just documents what each
    input tensor IS, what it MEANS, and what its SHAPE represents.
    
    INPUTS:
    -------
    routing_logits: shape (T, 256), dtype bfloat16
        Raw router output. For each of T tokens, a score for each of 256 experts.
        Higher score = router thinks this expert is more relevant for this token.
    
    routing_bias: shape (256,), dtype bfloat16
        Additive bias on routing scores. Used to balance expert load —
        if an expert is underutilized, its bias gets increased.
    
    hidden_states: shape (T, 7168), dtype float8_e4m3fn
        The token activations entering this MoE layer.
        Quantized to FP8 (4-bit mantissa, 3-bit exponent) for efficiency.
        Each value is approximate — the real value is hidden_states * scale.
    
    hidden_states_scale: shape (56, T), dtype float32
        Block scales for hidden_states. 7168 / 128 = 56 blocks per token.
        Scale layout: scale[block_idx, token_idx]
        Real value of hidden_states[token, col] = 
            hidden_states[token, col] * scale[col // 128, token]
    
    gemm1_weights: shape (32, 4096, 7168), dtype float8_e4m3fn
        Expert weights for the "up+gate" projection.
        32 local experts, each with a (4096, 7168) weight matrix.
        
        WHY 4096 and not 2048? Because SwiGLU needs TWO projections:
          - Rows [0:2048]    = W_gate (gate projection)
          - Rows [2048:4096] = W_up   (up projection)
        They're fused into one matrix for efficiency (one GEMM instead of two).
        
        The computation: [gate; up] = hidden_state @ W1^T
        gives a (T, 4096) result that gets split into two (T, 2048) halves.
    
    gemm1_weights_scale: shape (32, 32, 56), dtype float32
        Block scales for gemm1_weights.
        32 experts × (4096/128=32) N-blocks × (7168/128=56) K-blocks.
        Real value: W1[expert][n, k] = W1_fp8[expert][n, k] * scale[expert][n//128, k//128]
    
    gemm2_weights: shape (32, 7168, 2048), dtype float8_e4m3fn
        Expert weights for the "down" projection.
        Takes the (T, 2048) SwiGLU output back to (T, 7168).
    
    gemm2_weights_scale: shape (32, 56, 16), dtype float32
        Block scales for gemm2_weights.
        32 experts × (7168/128=56) N-blocks × (2048/128=16) K-blocks.
    
    local_expert_offset: int
        Global index of the first local expert. If this GPU has experts 64-95,
        then local_expert_offset = 64.
    
    routed_scaling_factor: float
        Multiplier on routing weights. Adjusts overall contribution of routed experts.
    
    output: shape (T, 7168), dtype bfloat16
        Pre-allocated output buffer. We write our result here.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════ #
#  STEP 1: Dequantize Hidden States (FP8 → FP32)                            #
# ═══════════════════════════════════════════════════════════════════════════ #
def dequantize_hidden_states(hidden_states_fp8, hidden_states_scale):
    """
    Convert FP8 hidden states back to full precision using block scales.
    
    FP8 E4M3 can represent values roughly in [-448, 448] with ~4 bits of
    precision. The block scale recovers the true magnitude.
    
    Args:
        hidden_states_fp8: (T, 7168) FP8 values (stored as float32 after cast)
        hidden_states_scale: (56, T) FP32 scales
    
    Returns:
        (T, 7168) FP32 dequantized hidden states
    
    Example for one token, one block:
        fp8_values = [0.5, -1.0, 0.75, ...]   # 128 values, block index 3
        scale = 0.0023                          # scale[3, token]
        real_values = [0.5*0.0023, -1.0*0.0023, 0.75*0.0023, ...]
    """
    T, H = hidden_states_fp8.shape  # T tokens, H=7168 hidden dim
    num_h_blocks = H // BLOCK_Q     # 7168 / 128 = 56 blocks

    # Reshape to expose the block structure:
    # (T, 7168) → (T, 56, 128) — 56 blocks of 128 values each
    x = hidden_states_fp8.reshape(T, num_h_blocks, BLOCK_Q)
    
    # Scale shape: (56, T) → transpose to (T, 56) → add dim for broadcast → (T, 56, 1)
    s = hidden_states_scale.T[:, :, np.newaxis]  # (T, 56, 1)
    
    # Multiply: (T, 56, 128) * (T, 56, 1) → (T, 56, 128)
    # Each block of 128 values shares one scale
    result = x * s
    
    # Reshape back: (T, 56, 128) → (T, 7168)
    return result.reshape(T, H)


# ═══════════════════════════════════════════════════════════════════════════ #
#  STEP 2: Routing — Decide Which Experts Process Which Tokens               #
# ═══════════════════════════════════════════════════════════════════════════ #
def compute_routing(routing_logits, routing_bias, routed_scaling_factor):
    """
    The routing algorithm decides, for each token, which 8 of 256 experts
    should process it, and with what weights.
    
    The algorithm has 3 stages:
      1. Score all experts (sigmoid + bias)
      2. Select top-4 GROUPS (coarse filtering)
      3. Select top-8 EXPERTS from those groups (fine selection)
    
    Args:
        routing_logits: (T, 256) raw router scores
        routing_bias: (256,) load-balancing bias
        routed_scaling_factor: float multiplier
    
    Returns:
        topk_idx: (T, 8) — which 8 experts each token selected
        topk_w:   (T, 8) — normalized weights for each selected expert
    """
    T = routing_logits.shape[0]
    
    # ─── Stage 2a: Score all experts ─────────────────────────────────────
    # Sigmoid squashes logits to [0, 1] — these are the "true" scores
    s = sigmoid(routing_logits)  # (T, 256) in [0, 1]
    
    # Add bias for load balancing (doesn't affect final weights, only selection)
    s_with_bias = s + routing_bias  # (T, 256)
    
    # ─── Stage 2b: Group-level selection ─────────────────────────────────
    # WHY GROUPS? With 256 experts, we want diversity. Without groups, all 8
    # selected experts might come from the same "specialty area." Groups force
    # the router to spread selections across different areas.
    
    # Reshape into 8 groups of 32 experts each:
    # (T, 256) → (T, 8, 32)
    s_grouped = s_with_bias.reshape(T, N_GROUP, GROUP_SIZE)
    
    # Score each group by its top-2 experts (proxy for group quality)
    # For each group, find the 2 highest-scoring experts, sum their scores
    # This asks: "How good are the BEST experts in this group?"
    group_top2 = np.sort(s_grouped, axis=2)[:, :, -2:]  # (T, 8, 2) — top 2 per group
    group_scores = group_top2.sum(axis=2)                 # (T, 8) — sum of top 2
    
    # Select top-4 groups (out of 8)
    # Each token picks the 4 groups with highest scores
    group_top_indices = np.argsort(group_scores, axis=1)[:, -TOPK_GROUP:]  # (T, 4)
    
    # Create a mask: which groups are selected? (T, 8) boolean
    group_mask = np.zeros((T, N_GROUP), dtype=bool)
    for t in range(T):
        group_mask[t, group_top_indices[t]] = True
    
    # ─── Stage 2c: Expert-level selection within selected groups ──────────
    # Expand group mask to expert mask:
    # group_mask (T, 8) → expert_mask (T, 256)
    # If group g is selected, experts g*32 to (g+1)*32-1 are eligible
    expert_mask = np.repeat(group_mask, GROUP_SIZE, axis=1)  # (T, 256)
    
    # Zero out experts in non-selected groups
    scores_pruned = np.where(expert_mask, s_with_bias, -np.inf)  # (T, 256)
    
    # Select top-8 experts from the eligible ~128 experts (4 groups × 32)
    topk_idx = np.argsort(scores_pruned, axis=1)[:, -TOP_K:]  # (T, 8)
    
    # ─── Stage 2d: Compute normalized routing weights ────────────────────
    # IMPORTANT: weights use the ORIGINAL sigmoid scores (without bias)!
    # Bias affects WHICH experts are selected, not HOW MUCH they contribute.
    topk_s = np.take_along_axis(s, topk_idx, axis=1)  # (T, 8) — sigmoid scores
    
    # Normalize: weights sum to 1 per token
    topk_w = topk_s / (topk_s.sum(axis=1, keepdims=True) + 1e-20)  # (T, 8)
    
    # Apply global scaling factor
    topk_w = topk_w * routed_scaling_factor  # (T, 8)
    
    return topk_idx, topk_w


# ═══════════════════════════════════════════════════════════════════════════ #
#  STEP 3: Dequantize Expert Weights (FP8 → FP32)                           #
# ═══════════════════════════════════════════════════════════════════════════ #
def dequantize_weight(w_fp8, scale, out_dim, in_dim):
    """
    Convert an FP8 weight matrix back to FP32 using block scales.
    
    The weight is divided into 128×128 blocks. Each block shares one scale.
    
    Args:
        w_fp8: (out_dim, in_dim) FP8 weight matrix
        scale: (out_dim//128, in_dim//128) FP32 block scales
        out_dim: number of output features (rows)
        in_dim: number of input features (cols)
    
    Returns:
        (out_dim, in_dim) FP32 dequantized weight
    
    Visual for a 384×256 weight with BLOCK_Q=128:
    
        in_dim=256 →  [block0] [block1]
                     ┌────────┬────────┐
        out_dim=384  │ s[0,0] │ s[0,1] │  ← 128 rows share scale s[0,:]
                     ├────────┼────────┤
                     │ s[1,0] │ s[1,1] │  ← next 128 rows
                     ├────────┼────────┤
                     │ s[2,0] │ s[2,1] │  ← last 128 rows
                     └────────┴────────┘
    """
    nb_out = out_dim // BLOCK_Q  # number of row-blocks
    nb_in = in_dim // BLOCK_Q    # number of col-blocks
    
    # Reshape weight to expose blocks: (nb_out, 128, nb_in, 128)
    w = w_fp8.reshape(nb_out, BLOCK_Q, nb_in, BLOCK_Q)
    
    # Reshape scale for broadcasting: (nb_out, 1, nb_in, 1)
    s = scale.reshape(nb_out, 1, nb_in, 1)
    
    # Multiply: each 128×128 block × its scalar scale
    result = w * s  # (nb_out, 128, nb_in, 128)
    
    # Reshape back: (out_dim, in_dim)
    return result.reshape(out_dim, in_dim)


# ═══════════════════════════════════════════════════════════════════════════ #
#  STEP 4: The Expert FFN — GEMM1 → SwiGLU → GEMM2                          #
# ═══════════════════════════════════════════════════════════════════════════ #
def expert_ffn(tokens, w1, w1_scale, w2, w2_scale):
    """
    One expert's feed-forward network.
    
    Architecture:
        tokens (Tk, 7168)
           │
           ▼
        GEMM1: tokens @ W1^T → (Tk, 4096)
           │
           ├── split ──┐
           ▼            ▼
        gate (Tk,2048)  up (Tk,2048)
           │            │
           │            ▼
           │         silu(up)
           │            │
           ▼            ▼
        gate    ×    silu(up)     ← this is SwiGLU
           │
           ▼
        intermediate (Tk, 2048)
           │
           ▼
        GEMM2: intermediate @ W2^T → (Tk, 7168)
           │
           ▼
        output (Tk, 7168)
    
    Args:
        tokens: (Tk, 7168) FP32 — this expert's assigned tokens
        w1: (4096, 7168) FP8 — fused gate+up weight
        w1_scale: (32, 56) FP32 — block scales for w1
        w2: (7168, 2048) FP8 — down projection weight
        w2_scale: (56, 16) FP32 — block scales for w2
    
    Returns:
        (Tk, 7168) FP32 — expert output
    """
    Tk = tokens.shape[0]
    
    # ─── GEMM1: Up+Gate projection ──────────────────────────────────────
    # Dequantize W1: (4096, 7168) FP8 → FP32
    W1 = dequantize_weight(w1, w1_scale, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    # W1 is (4096, 7168)
    
    # Matrix multiply: (Tk, 7168) @ (7168, 4096) = (Tk, 4096)
    g1 = tokens @ W1.T  # (Tk, 4096)
    
    # ─── SwiGLU Activation ──────────────────────────────────────────────
    # Split the 4096-dim output into two 2048-dim halves
    gate = g1[:, :INTERMEDIATE_SIZE]       # (Tk, 2048) — first half
    up   = g1[:, INTERMEDIATE_SIZE:]       # (Tk, 2048) — second half
    
    # SwiGLU: gate * silu(up)
    # SiLU(x) = x * sigmoid(x)   (also called "Swish")
    #
    # WHY SwiGLU? It's empirically better than ReLU for LLMs.
    # The "gate" learns to control information flow (like LSTM gates).
    # The "up" projection provides the actual features.
    # Multiplying them lets the network learn: "I want feature X 
    # but only when condition Y is true."
    intermediate = gate * silu(up)  # (Tk, 2048)
    
    # ─── GEMM2: Down projection ────────────────────────────────────────
    # Dequantize W2: (7168, 2048) FP8 → FP32
    W2 = dequantize_weight(w2, w2_scale, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    # W2 is (7168, 2048)
    
    # Matrix multiply: (Tk, 2048) @ (2048, 7168) = (Tk, 7168)
    output = intermediate @ W2.T  # (Tk, 7168)
    
    return output


# ═══════════════════════════════════════════════════════════════════════════ #
#  STEP 5: Dispatch + Gather — Sort Tokens by Expert, Run, Scatter Back      #
# ═══════════════════════════════════════════════════════════════════════════ #
def dispatch_and_compute(
    hidden_fp32,          # (T, 7168) FP32 dequantized hidden states
    topk_idx,             # (T, 8) which experts each token selected
    topk_w,               # (T, 8) routing weights
    gemm1_weights,        # (32, 4096, 7168) FP8
    gemm1_weights_scale,  # (32, 32, 56) FP32
    gemm2_weights,        # (32, 7168, 2048) FP8
    gemm2_weights_scale,  # (32, 56, 16) FP32
    local_expert_offset,  # int
):
    """
    For each local expert:
      1. Gather all tokens assigned to it
      2. Run the expert FFN
      3. Multiply by routing weights
      4. Scatter results back, accumulating overlapping tokens
    
    TOKEN FLOW EXAMPLE (T=4 tokens, 3 local experts):
    
    topk_idx (showing only local experts, offset subtracted):
        Token 0: experts [0, 2, ...]     Token 0 goes to expert 0 and 2
        Token 1: experts [1, 2, ...]     Token 1 goes to expert 1 and 2
        Token 2: experts [0, ...]        Token 2 goes to expert 0
        Token 3: experts [1, ...]        Token 3 goes to expert 1
    
    Expert 0 processes: [Token 0, Token 2]  → 2 tokens
    Expert 1 processes: [Token 1, Token 3]  → 2 tokens
    Expert 2 processes: [Token 0, Token 1]  → 2 tokens
    
    Note: Token 0 appears in BOTH expert 0 and expert 2!
    Its final output = w0 * Expert0(token0) + w2 * Expert2(token0)
    This is why we ACCUMULATE (add) results, not overwrite.
    
    Returns:
        (T, 7168) FP32 accumulated output
    """
    T = hidden_fp32.shape[0]
    accum = np.zeros((T, HIDDEN_SIZE), dtype=np.float32)
    
    # Convert global expert indices to local (0-31)
    local_idx = topk_idx - local_expert_offset  # (T, 8)
    
    # For each of the 32 local experts
    for expert_local_id in range(NUM_LOCAL_EXPERTS):
        
        # ─── GATHER: Find all (token, slot) pairs assigned to this expert ──
        # A "slot" is which of the 8 top-k positions this expert occupies
        token_indices = []
        slot_indices = []
        
        for t in range(T):
            for k in range(TOP_K):
                if local_idx[t, k] == expert_local_id:
                    token_indices.append(t)
                    slot_indices.append(k)
        
        if len(token_indices) == 0:
            continue  # No tokens for this expert
        
        token_indices = np.array(token_indices)
        slot_indices = np.array(slot_indices)
        Tk = len(token_indices)  # Number of tokens for this expert
        
        print(f"  Expert {expert_local_id}: processing {Tk} tokens")
        
        # ─── Gather the tokens for this expert ──────────────────────────
        # Pull out the Tk rows from the full (T, 7168) activation matrix
        expert_tokens = hidden_fp32[token_indices]  # (Tk, 7168)
        
        # ─── RUN THE EXPERT FFN ─────────────────────────────────────────
        expert_output = expert_ffn(
            expert_tokens,
            gemm1_weights[expert_local_id],
            gemm1_weights_scale[expert_local_id],
            gemm2_weights[expert_local_id],
            gemm2_weights_scale[expert_local_id],
        )  # (Tk, 7168)
        
        # ─── Get routing weights for these (token, slot) pairs ──────────
        weights = topk_w[token_indices, slot_indices]  # (Tk,)
        
        # ─── SCATTER: Weighted accumulate back to original positions ────
        # Each expert output row gets multiplied by its routing weight,
        # then ADDED to the corresponding token's accumulator.
        #
        # The same token can receive contributions from multiple experts.
        # Example: Token 0 gets:
        #   accum[0] += weight_expert0 * expert0_output
        #   accum[0] += weight_expert2 * expert2_output
        weighted_output = expert_output * weights[:, np.newaxis]  # (Tk, 7168)
        
        for i in range(Tk):
            accum[token_indices[i]] += weighted_output[i]
    
    return accum


# ═══════════════════════════════════════════════════════════════════════════ #
#  COMPLETE MoE KERNEL — Putting It All Together                             #
# ═══════════════════════════════════════════════════════════════════════════ #
def moe_kernel_numpy(
    routing_logits,         # (T, 256) float32
    routing_bias,           # (256,) float32
    hidden_states,          # (T, 7168) float32 (representing FP8)
    hidden_states_scale,    # (56, T) float32
    gemm1_weights,          # (32, 4096, 7168) float32 (representing FP8)
    gemm1_weights_scale,    # (32, 32, 56) float32
    gemm2_weights,          # (32, 7168, 2048) float32 (representing FP8)
    gemm2_weights_scale,    # (32, 56, 16) float32
    local_expert_offset,    # int
    routed_scaling_factor,  # float
):
    """
    Complete MoE layer forward pass.
    
    FULL PIPELINE:
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │  INPUT: hidden_states (T, 7168) FP8                                 │
    │         routing_logits (T, 256)                                      │
    │                                                                      │
    │  ┌─────────────────────┐    ┌─────────────────────────────┐         │
    │  │ 1. DEQUANT HIDDEN   │    │ 2. ROUTING                   │         │
    │  │    FP8 → FP32       │    │    logits → expert selection  │         │
    │  │    (T,7168) × scale  │    │    → top-8 experts + weights │         │
    │  └─────────┬───────────┘    └──────────────┬──────────────┘         │
    │            │                                │                        │
    │            ▼                                ▼                        │
    │  ┌─────────────────────────────────────────────────────────┐        │
    │  │ 3. DISPATCH: For each expert, gather assigned tokens     │        │
    │  │                                                          │        │
    │  │    Expert 0: [token 3, token 7, token 12]                │        │
    │  │    Expert 1: [token 1, token 5]                          │        │
    │  │    Expert 2: [token 3, token 5, token 8, token 12]       │        │
    │  │    ...                                                   │        │
    │  │                                                          │        │
    │  │ 4. Per-expert FFN:                                       │        │
    │  │    tokens ──→ GEMM1 ──→ SwiGLU ──→ GEMM2 ──→ output    │        │
    │  │    (Tk,7168)  (Tk,4096)  (Tk,2048)  (Tk,7168)           │        │
    │  │                                                          │        │
    │  │ 5. SCATTER: weighted accumulate back                     │        │
    │  │    output[token] += weight × expert_result               │        │
    │  └─────────────────────────────────────────────────────────┘        │
    │                                                                      │
    │  OUTPUT: (T, 7168) BF16                                             │
    └──────────────────────────────────────────────────────────────────────┘
    
    WHY IS THIS HARD TO OPTIMIZE?
    
    1. LOAD IMBALANCE: Expert 0 might get 500 tokens, Expert 1 gets 3.
       GPUs hate variable-size workloads.
    
    2. GATHER/SCATTER: Pulling tokens from random rows (gather) and 
       writing back to random rows (scatter) is memory-unfriendly.
       GPUs prefer sequential, coalesced memory access.
    
    3. FP8 DEQUANTIZATION: Can't just call standard BLAS — need custom
       kernels that dequant inside the GEMM inner loop.
    
    4. KERNEL LAUNCH OVERHEAD: 32 experts × 3 ops = 96 kernel launches.
       Each launch has ~5μs overhead → 480μs just in overhead.
    
    5. TILE WASTE: GPU tiles (e.g. 64×128) process tokens in blocks.
       Expert with 5 tokens wastes 92% of a 64-row tile.
    """
    T = routing_logits.shape[0]
    print(f"MoE forward: T={T} tokens, {NUM_LOCAL_EXPERTS} local experts")
    print(f"  Hidden size: {HIDDEN_SIZE}, Intermediate: {INTERMEDIATE_SIZE}")
    print(f"  Top-K: {TOP_K}, Groups: {N_GROUP}, TopK-groups: {TOPK_GROUP}")
    print()
    
    # ─── Step 1: Dequantize hidden states ────────────────────────────────
    print("Step 1: Dequantizing hidden states (FP8 → FP32)...")
    hidden_fp32 = dequantize_hidden_states(hidden_states, hidden_states_scale)
    print(f"  Input:  ({T}, {HIDDEN_SIZE}) FP8")
    print(f"  Output: ({T}, {HIDDEN_SIZE}) FP32")
    print(f"  Scale blocks: {HIDDEN_SIZE // BLOCK_Q} blocks of {BLOCK_Q} values")
    print()
    
    # ─── Step 2: Routing ─────────────────────────────────────────────────
    print("Step 2: Computing routing decisions...")
    topk_idx, topk_w = compute_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )
    print(f"  Each token selects {TOP_K} experts from {NUM_EXPERTS} total")
    print(f"  topk_idx shape: {topk_idx.shape}  (which experts)")
    print(f"  topk_w shape:   {topk_w.shape}  (how much weight)")
    print(f"  Example token 0:")
    print(f"    Selected experts: {topk_idx[0]}")
    print(f"    Weights:          {np.round(topk_w[0], 4)}")
    print()
    
    # ─── Step 3-5: Dispatch, compute, scatter ────────────────────────────
    print("Steps 3-5: Dispatch → Expert FFN → Scatter...")
    local_start = local_expert_offset
    local_end = local_expert_offset + NUM_LOCAL_EXPERTS
    
    # Count how many tokens go to local vs remote experts
    local_hits = ((topk_idx >= local_start) & (topk_idx < local_end)).sum()
    total_hits = topk_idx.size
    print(f"  Token-expert pairs: {total_hits} total, {local_hits} local")
    print()
    
    accum = dispatch_and_compute(
        hidden_fp32, topk_idx, topk_w,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        local_expert_offset,
    )
    
    print()
    print(f"Output shape: ({T}, {HIDDEN_SIZE}) → convert to BF16")
    
    return accum  # In real code, this gets cast to BF16


# ═══════════════════════════════════════════════════════════════════════════ #
#  DEMO: Run with small random inputs                                        #
# ═══════════════════════════════════════════════════════════════════════════ #
if __name__ == "__main__":
    np.random.seed(42)
    
    # Use small sizes for demo (real sizes commented)
    T = 4  # real: hundreds to thousands
    
    # Simulate inputs (in reality these come from previous transformer layer)
    routing_logits = np.random.randn(T, NUM_EXPERTS).astype(np.float32)
    routing_bias = np.random.randn(NUM_EXPERTS).astype(np.float32) * 0.1
    
    # Simulate FP8 hidden states (values in small range, like real FP8)
    hidden_states = (np.random.randn(T, HIDDEN_SIZE) * 0.5).astype(np.float32)
    hidden_states_scale = (np.random.rand(HIDDEN_SIZE // BLOCK_Q, T) * 0.01 + 0.001).astype(np.float32)
    
    # Simulate FP8 expert weights
    gemm1_weights = (np.random.randn(NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE) * 0.1).astype(np.float32)
    gemm1_weights_scale = (np.random.rand(NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE // BLOCK_Q, HIDDEN_SIZE // BLOCK_Q) * 0.01).astype(np.float32)
    
    gemm2_weights = (np.random.randn(NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE) * 0.1).astype(np.float32)
    gemm2_weights_scale = (np.random.rand(NUM_LOCAL_EXPERTS, HIDDEN_SIZE // BLOCK_Q, INTERMEDIATE_SIZE // BLOCK_Q) * 0.01).astype(np.float32)
    
    local_expert_offset = 64  # This GPU has experts 64-95
    routed_scaling_factor = 1.0
    
    # Run!
    output = moe_kernel_numpy(
        routing_logits, routing_bias,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor,
    )
    
    print(f"\nFinal output range: [{output.min():.6f}, {output.max():.6f}]")
    print(f"Final output norm:  {np.linalg.norm(output):.6f}")