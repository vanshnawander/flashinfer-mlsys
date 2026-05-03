# MoE Dequant + Routing + Local Expert Compute: Detailed Walkthrough

This document explains the exact code path in your snippet, with tensor shapes and why each method is used.

The three stages are:

1. FP8 block-scale dequantization (inputs + weights)
2. No-aux routing (DeepSeek-style grouped top-k)
3. Local expert compute and accumulation

---

## Why this code feels hard to read

This part is tricky because several things happen at once:

- Quantized tensors are stored in one shape, but scales are stored in a different blocked shape.
- Routing uses two different scores (`s` and `s_with_bias`) for two different purposes.
- Many ops are shape transforms only (`permute`, `unsqueeze`, `reshape`, `repeat_interleave`) and do not change values, but they are critical for correctness.
- The code mixes global expert space (`E_global=256`) and local expert space (`E_local=32`).

If you keep two mental models, it becomes easier:

- Model 1: "What is the math?"
  - `real_value ~= fp8_value * scale`
  - choose experts by top-k routing
  - run local experts and weighted sum outputs
- Model 2: "What is the memory layout?"
  - where scales live, and how they are expanded to match data tensors

---

## Stage 1: FP8 block-scale dequantization

### 1.1 Hidden states dequant

Code:

```python
A_fp32 = hidden_states.to(torch.float32)
A_scale = hidden_states_scale.to(torch.float32)                # [H/128, T]
A_scale_TH = A_scale.permute(1, 0).contiguous()                # [T, H/128]
A_scale_expanded = (
    A_scale_TH.unsqueeze(-1)
    .repeat(1, 1, BLOCK)                                        # [T, H/128, 128]
    .reshape(T, H)                                              # [T, H]
    .contiguous()
)
A = A_fp32 * A_scale_expanded                                   # [T, H]
```

What it means:

- `hidden_states` is FP8 with shape `[T, H]`.
- `hidden_states_scale` is block scale with shape `[H/128, T]`.
- Each scale value applies to a block of 128 hidden dimensions for one token.

Why each method is used:

- `to(torch.float32)`:
  - FP8 is too low precision for stable compute.
  - Convert to FP32 before multiplying by scales and before GEMM.

- `permute(1, 0)`:
  - Original scale layout is `[H/128, T]`.
  - Hidden states are indexed naturally as `[T, H]`.
  - Permuting gives `[T, H/128]`, making token dimension first.

- `contiguous()`:
  - `permute` often returns a non-contiguous view.
  - Later `reshape` and kernels are safer/faster with contiguous memory.

- `unsqueeze(-1)`:
  - `[T, H/128] -> [T, H/128, 1]`.
  - Adds a singleton dimension so each block scale can be repeated across 128 elements.

- `repeat(1, 1, BLOCK)`:
  - Expands each block scale from 1 element to 128 elements.
  - `[T, H/128, 1] -> [T, H/128, 128]`.

- `reshape(T, H)`:
  - Flattens blocked hidden axis (`H/128 * 128`) back to `H`.
  - Gives one scale value per hidden element.

- Elementwise multiply:
  - `A = A_fp32 * A_scale_expanded`.
  - Now `A` is dequantized hidden states in FP32.

Equivalent formula:

- For token `t`, hidden dim `h`:
  - `A[t, h] = float(hidden_states[t, h]) * hidden_states_scale[h // 128, t]`

---

### 1.2 GEMM1 weights dequant (`W13`)

Code:

```python
W13_fp32 = gemm1_weights.to(torch.float32)
S13 = gemm1_weights_scale.to(torch.float32)
S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)  # [E, 2I, H/128]
S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)  # [E, 2I, H]
W13 = W13_fp32 * S13_expanded                              # [E, 2I, H]
```

Shapes:

- `gemm1_weights`: `[E_local, 2I, H]` (FP8)
- `gemm1_weights_scale`: `[E_local, (2I)/128, H/128]` (FP32)

What `repeat_interleave` does:

- `repeat_interleave(x, BLOCK, dim=1)`:
  - repeats each block-scale row across 128 output channels (`2I` axis).
- `repeat_interleave(..., BLOCK, dim=2)`:
  - repeats each block-scale column across 128 hidden channels (`H` axis).

Why used:

- Scale tensor is block-indexed, weight tensor is element-indexed.
- Interleave expands block scales to per-element scales matching weight shape.

Equivalent formula:

- For expert `e`, out dim `o`, in dim `h`:
  - `W13[e, o, h] = float(gemm1_weights[e, o, h]) * gemm1_weights_scale[e, o//128, h//128]`

---

### 1.3 GEMM2 weights dequant (`W2`)

Code:

```python
W2_fp32 = gemm2_weights.to(torch.float32)
S2 = gemm2_weights_scale.to(torch.float32)
S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)    # [E, H, I/128]
S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)    # [E, H, I]
W2 = W2_fp32 * S2_expanded                                 # [E, H, I]
```

Same idea as `W13`:

- `gemm2_weights`: `[E_local, H, I]` FP8
- `gemm2_weights_scale`: `[E_local, H/128, I/128]` FP32
- Expand scales to full `[E_local, H, I]` and multiply.

---

## Stage 2: No-aux routing

### 2.1 Convert logits to probabilities

Code:

```python
logits = routing_logits.to(torch.float32)   # [T, E]
bias = routing_bias.to(torch.float32).reshape(-1)  # [E]
s = 1.0 / (1.0 + torch.exp(-logits))        # sigmoid, [T, E]
s_with_bias = s + bias                      # [T, E]
```

Why:

- `s` is expert affinity per token.
- Bias helps selection behavior (`s_with_bias`) but is not used for final weight normalization.

`reshape(-1)`:

- Flattens bias to 1D `[E]`.
- Makes broadcast with `[T, E]` explicit and safe.

---

### 2.2 Group experts and score groups

Code:

```python
group_size = E_global // N_GROUP  # 32
s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)  # [T, 8, 32]
top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
group_scores = top2_vals.sum(dim=2)  # [T, 8]
```

Why this grouping exists:

- Instead of top-k over all 256 experts directly, it first selects best groups.
- This limits routing spread and makes dispatch more structured.

Method notes:

- `view(T, 8, 32)`:
  - Reinterprets expert axis into groups.
  - No data copy if contiguous.
- `topk(..., dim=2, k=2)`:
  - per group, get top-2 expert scores.
- `sum(dim=2)`:
  - group score = sum of top-2 experts in group.

---

### 2.3 Select top groups and mask experts

Code:

```python
_, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)  # [T, 4]
group_mask = torch.zeros_like(group_scores)  # [T, 8]
group_mask.scatter_(1, group_idx, 1.0)
score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)  # [T, E]
```

What each op does:

- `zeros_like(group_scores)`:
  - create mask initialized with 0.

- `scatter_(dim=1, index=group_idx, value=1.0)`:
  - in each token row, put 1 at selected group indices.
  - in-place write, avoids loop.

- `unsqueeze(2)`:
  - `[T, 8] -> [T, 8, 1]`.

- `expand(T, 8, 32)`:
  - broadcast each group mask value to all 32 experts in that group.
  - unlike `repeat`, `expand` is usually view-like (no full copy).

- `reshape(T, E_global)`:
  - flatten grouped expert mask back to expert axis `[T, 256]`.

---

### 2.4 Global top-k within selected groups

Code:

```python
neg_inf = torch.finfo(torch.float32).min
scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
_, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)  # [T, 8]
```

Why:

- `masked_fill` sets forbidden experts to very negative values.
- Then global top-k over all experts effectively chooses only among allowed groups.

Method:

- `masked_fill(mask, value)`:
  - returns tensor where masked positions are replaced.

---

### 2.5 Build normalized routing weights

Code:

```python
M = torch.zeros_like(s)                 # [T, E]
M.scatter_(1, topk_idx, 1.0)            # top-k indicator
weights = s * M                         # keep only selected experts
weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
weights = (weights / weights_sum) * routed_scaling_factor
```

Important detail:

- Selection used `s_with_bias`, but weight normalization uses `s` (without bias).
- This is intentional in DeepSeek no-aux routing.

Method notes:

- `keepdim=True` keeps shape `[T, 1]` for broadcast division over `[T, E]`.
- `1e-20` avoids divide-by-zero edge cases.

---

## Stage 3: Local expert compute and accumulation

Code starts with:

```python
output = torch.zeros((T, H), dtype=torch.float32, device=device)
local_start = int(local_expert_offset)
```

What this means:

- Output buffer accumulates contributions from local experts only.
- `local_expert_offset` maps local expert index (`0..E_local-1`) to global expert id.

Conceptually per local expert:

1. Find tokens that selected this global expert in top-k.
2. Gather those token hidden states.
3. Run GEMM1 -> split -> SwiGLU -> GEMM2.
4. Multiply by token route weights for that expert.
5. Scatter-add into `output` rows.

Why this is hard:

- Token sets differ per expert, so compute is ragged and dynamic.
- Needs gather/scatter (not just dense matmul).
- Correctness depends on global/local expert id mapping and matching weights.

---

## Quick glossary of methods used

- `to(dtype)`:
  - Cast tensor dtype.

- `permute(...)`:
  - Reorder dimensions.

- `contiguous()`:
  - Materialize contiguous memory layout.

- `unsqueeze(dim)`:
  - Insert size-1 dimension.

- `repeat(...)`:
  - Physically replicate data along dimensions.

- `repeat_interleave(..., dim=...)`:
  - Repeat each element/chunk along a dimension.

- `reshape(...)`:
  - Change shape (copy only if needed).

- `view(...)`:
  - Reshape as a view (requires compatible contiguous layout).

- `topk(...)`:
  - Select k largest/smallest elements along a dimension.

- `scatter_`:
  - In-place write values at indices along a dimension.

- `expand(...)`:
  - Broadcast size-1 dimensions without full data copy.

- `masked_fill(mask, value)`:
  - Replace masked elements with value.

- `sum(dim=..., keepdim=...)`:
  - Reduce along dimension; optionally retain dimension.

---

## Performance note (important)

This reference-style code is very clear for correctness but expensive:

- Expands block scales to full dense tensors (`A_scale_expanded`, `S13_expanded`, `S2_expanded`).
- Materializes full FP32 weights from FP8.
- Uses scatter/gather heavy logic.

Optimized kernels usually do:

- On-the-fly dequant per tile (avoid full expanded scales/weights),
- fused routing + dispatch where possible,
- fused GEMM epilogues for activation/accumulation.

