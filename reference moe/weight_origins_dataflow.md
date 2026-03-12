# MoE Architecture — Weight Origins & Data Flow

> Quick reference: where every input comes from, what W1/W3/W2 are,
> and how the transformer pipeline feeds into our kernel.

---

## The Transformer Block

A single DeepSeek-V3 layer looks like this:

```
Input x [T, H=7168]
    │
    ├── LayerNorm
    ├── Self-Attention (Q, K, V, softmax, output projection)
    ├── + residual
    │
    ├── LayerNorm
    ├── Quantize to FP8 + compute block scales
    ├── Router: x @ gate_weight.T → routing_logits [T, 256]
    │
    ├── ★ OUR KERNEL (MoE FFN) ★
    │      → output [T, H] BF16
    │
    ├── + residual
    │
Output x [T, H=7168]
```

**Our kernel replaces the standard FFN layer.** We receive the output of
attention + layernorm as FP8 quantized hidden states, and produce BF16 output.

---

## Why Three Weight Matrices (W1, W3, W2)?

### Normal FFN (GPT-2 era):
```
x ──→ W_up [H, 4H] ──→ ReLU ──→ W_down [4H, H] ──→ y
       "expand"                    "compress"
```
2 matrices: one expands, one compresses.

### SwiGLU FFN (LLaMA / DeepSeek-V3):
```
x ──┬──→ W1 (up)   [H → I]  ──→ pass through ──┐
    │                                             ├─→ element multiply ──→ c [I]
    └──→ W3 (gate)  [H → I]  ──→ silu()    ──────┘
                                                         │
                                                    W2 (down) [I → H]
                                                         │
                                                         ▼
                                                     y [H]
```
3 matrices. The "gate" controls which dimensions survive.

### The three roles:

| Matrix | Name | Shape (per expert) | Role |
|--------|------|-------------------|------|
| **W1** | Up projection | `[I, H]` = `[2048, 7168]` | Linearly projects to intermediate space |
| **W3** | Gate projection | `[I, H]` = `[2048, 7168]` | Produces gate values (passed through silu) |
| **W2** | Down projection | `[H, I]` = `[7168, 2048]` | Projects back to hidden dimension |

### Mathematical formulation:
```
y = W2ᵀ · ( (W1ᵀ · x) ⊙ silu(W3ᵀ · x) )
         ─────────────────────────────────
              SwiGLU activation
```

Where:
- `⊙` = element-wise multiply
- `silu(z) = z × σ(z) = z / (1 + exp(-z))`

---

## Why W1 and W3 Are Concatenated

Doing two separate matmuls would waste launch overhead:
```python
# Slow: 2 kernel launches
up   = x @ W1.T    # [Tk, 2048]
gate = x @ W3.T    # [Tk, 2048]
```

Instead, stack them into one matrix and do one matmul:
```python
# Fast: 1 kernel launch
W13 = cat([W1, W3], dim=0)    # [4096, 7168]
g1  = x @ W13.T               # [Tk, 4096]

# Split the result:
up   = g1[:, 0:2048]          # first half = W1 output
gate = g1[:, 2048:4096]       # second half = W3 output
```

**This is why `gemm1_weights` has shape `[E, 4096, H]` = `[32, 4096, 7168]`.**
The 4096 dimension is `2 × I = 2 × 2048`.

---

## Where Weights Come From (Model Checkpoint → Our Kernel)

```
DeepSeek-V3 model checkpoint (on disk)
    │
    ├── layer_N/
    │     ├── attention/
    │     │     ├── q_proj.weight          ← NOT our concern
    │     │     ├── k_proj.weight
    │     │     ├── v_proj.weight
    │     │     └── o_proj.weight
    │     │
    │     └── moe/
    │           ├── gate.weight [H, E_global] → used to compute routing_logits
    │           ├── gate.bias   [E_global]    → routing_bias
    │           │
    │           ├── experts.0.w1.weight [I, H]  ─┐
    │           ├── experts.0.w3.weight [I, H]  ─┤→ gemm1_weights[0] = cat([w1, w3])
    │           ├── experts.0.w2.weight [H, I]  ─→ gemm2_weights[0]
    │           │
    │           ├── experts.1.w1.weight [I, H]  ─┐
    │           ├── experts.1.w3.weight [I, H]  ─┤→ gemm1_weights[1] = cat([w1, w3])
    │           ├── experts.1.w2.weight [H, I]  ─→ gemm2_weights[1]
    │           │
    │           └── ... (256 experts total, 32 local to this GPU)
    │
    ▼
  Benchmark Framework (flashinfer-bench)
    │
    ├── 1. Load FP32/BF16 weights
    ├── 2. Quantize to FP8 (float8_e4m3fn)
    ├── 3. Compute block scales: one FP32 scale per 128-element block
    ├── 4. Stack into tensors:
    │       gemm1_weights      [32, 4096, 7168] FP8
    │       gemm1_weights_scale [32, 32, 56]    FP32
    │       gemm2_weights      [32, 7168, 2048] FP8
    │       gemm2_weights_scale [32, 56, 16]    FP32
    │
    └── 5. Call our kernel(...)
```

---

## Every Input Tensor — Full Trace

### `routing_logits [T, 256] FP32`
```
Origin:   x @ gate_weight.T (linear layer, no activation)
Meaning:  Raw score for each of 256 experts, per token
Example:  token #5 might have high logits for experts 42, 87, 191...
Our use:  sigmoid → bias → group selection → top-8 expert picking
```

### `routing_bias [256] BF16`
```
Origin:   gate.bias from checkpoint (learned parameter)
Meaning:  Per-expert offset that nudges load balancing
Example:  bias[42] = 0.1 makes expert 42 more likely to be selected
Our use:  Added to sigmoid scores BEFORE expert selection
          NOT used for weight normalization (critical detail!)
```

### `hidden_states [T, 7168] FP8`
```
Origin:   LayerNorm(attention_output + residual), then quantized to FP8
Meaning:  The actual token representations we need to process
Example:  Each row is one token's 7168-dim feature vector
Our use:  Dequantize to FP32 → feed into GEMM1 for each expert
```

### `hidden_states_scale [56, T] FP32`
```
Origin:   Computed during FP8 quantization of hidden_states
Meaning:  One scale per 128-element block — scale[h÷128, token]
Layout:   TRANSPOSED! Row = hidden block (0..55), Column = token
Why 56:   7168 ÷ 128 = 56 blocks in hidden dimension
Our use:  output[t,h] = fp8_value[t,h] × scale[h÷128, t]
```

### `gemm1_weights [32, 4096, 7168] FP8`
```
Origin:   cat([expert.w1.weight, expert.w3.weight], dim=0), quantized
Meaning:  Combined up+gate projection for each local expert
Layout:   [expert_id, output_dim, input_dim]
          First 2048 output rows = W1 (up projection)
          Last  2048 output rows = W3 (gate projection)
Size:     32 × 4096 × 7168 × 1B = 896 MB total (28 MB per expert)
Our use:  g1 = hidden @ W13.T → [Tk, 4096], then split for SwiGLU
```

### `gemm1_weights_scale [32, 32, 56] FP32`
```
Origin:   Block scales for gemm1_weights (one scale per 128×128 block)
Layout:   [expert_id, output_blocks, input_blocks]
          32 = 4096 ÷ 128 output blocks
          56 = 7168 ÷ 128 input blocks
Size:     32 × 32 × 56 × 4B = 224 KB (negligible)
Our use:  dequant: W_fp32[n,k] = W_fp8[n,k] × scale[n÷128, k÷128]
```

### `gemm2_weights [32, 7168, 2048] FP8`
```
Origin:   expert.w2.weight, quantized to FP8
Meaning:  Down projection for each local expert
Layout:   [expert_id, output_dim, input_dim]
Size:     32 × 7168 × 2048 × 1B = 448 MB total (14 MB per expert)
Our use:  output = activated @ W2.T → [Tk, 7168]
```

### `gemm2_weights_scale [32, 56, 16] FP32`
```
Origin:   Block scales for gemm2_weights
Layout:   [expert_id, output_blocks, input_blocks]
          56 = 7168 ÷ 128 output blocks
          16 = 2048 ÷ 128 input blocks
Size:     32 × 56 × 16 × 4B = 112 KB (negligible)
```

### `local_expert_offset` (int)
```
Origin:   Framework tells us which shard of experts we own
Meaning:  Global ID of local expert 0
Example:  If offset=64, then our local expert 0 = global expert 64
          Our local expert 31 = global expert 95
Our use:  local_idx = topk_idx - offset (map global → local)
```

### `routed_scaling_factor` (float)
```
Origin:   Model hyperparameter (set during training)
Meaning:  Global multiplier on routing weights
Our use:  topk_w *= routed_scaling_factor (after normalization)
```

### `output [T, 7168] BF16` — DPS
```
Origin:   Pre-allocated by benchmark framework
Meaning:  Where we MUST write our result (destination passing style)
Our use:  output.copy_(accum.to(bfloat16)) at the very end
```

---

## Memory Budget Summary

| Category | Size | % of Total | Read Frequency |
|----------|------|-----------|----------------|
| W13 weights (FP8) | 896 MB | 66% | Once per expert |
| W2 weights (FP8) | 448 MB | 33% | Once per expert |
| Hidden states (FP8) | T × 7 KB | <1% | Once total |
| All scales | 336 KB | <0.1% | With weights |
| Routing data | T × 1 KB | <1% | Once total |
| **Total per forward** | **~1.34 GB** | 100% | — |

→ **99% of memory traffic is weight loading.**
→ Any optimization that reduces weight reads or overlaps them with compute wins big.

---

## Optimization Implications

| Insight | Optimization |
|---------|-------------|
| W1 and W3 are already concatenated | GEMM1 is a single matmul — good |
| Each expert has completely independent weights | Can process experts in parallel (grouped GEMM) |
| Same hidden states used for all experts | Hidden states are small, cacheable in L2 |
| Weights are 99% of memory traffic | Fused dequant avoids 4× materialization cost |
| Weights are FP8 | B200 has 4 PFLOPS FP8 tensor cores (vs 60 TFLOPS FP32) |
| 42 MB per expert fits in B200's 126 MB L2 | Can keep one expert's weights hot in L2 |
| Sequential expert loop wastes SM parallelism | Grouped/persistent kernel can use all 148 SMs |

---

> **Last updated:** 2026-03-13 | DeepSeek-V3 MoE architecture reference
