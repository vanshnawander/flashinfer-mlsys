# Kernel Operations — Plain English Breakdown

> Every single operation in `kernel.py` explained simply.
> No jargon — just what happens, what shape data is, and what math runs.

---

## The Big Picture

This kernel takes a batch of **T tokens** (each a 7168-dim vector in FP8) and routes them
through **32 local experts** (out of 256 total). Each expert is a mini neural network:
multiply → activate → multiply → output. The results from all experts are weighted
and summed per token.

```
Input tokens [T, 7168] FP8
        ↓
   ┌────────────┐
   │  DEQUANT   │ ← multiply by scales to get real FP32 values
   └────────────┘
        ↓
    [T, 7168] FP32
        ↓
   ┌────────────┐
   │  ROUTING   │ ← decide which 8 of 256 experts each token uses
   └────────────┘
        ↓
    topk_idx [T, 8]   (which experts)
    topk_w   [T, 8]   (how much weight each expert gets)
        ↓
   ┌────────────────────────────────────────────┐
   │  FOR EACH LOCAL EXPERT (0..31):            │
   │                                            │
   │  1. Gather tokens assigned to this expert  │
   │  2. GEMM1: tokens × W13  → [Tk, 4096]     │
   │  3. SwiGLU: activate     → [Tk, 2048]     │
   │  4. GEMM2: activated × W2 → [Tk, 7168]    │
   │  5. Multiply by routing weight             │
   │  6. Add to accumulator                     │
   └────────────────────────────────────────────┘
        ↓
    accum [T, 7168] FP32
        ↓
   ┌────────────┐
   │  CAST BF16 │ ← convert to bfloat16
   └────────────┘
        ↓
    output [T, 7168] BF16  (written into pre-allocated buffer)
```

---

## What Comes In (Kernel Inputs)

```
routing_logits       [T, 256]        FP32    raw router scores (before sigmoid)
routing_bias         [256]           BF16    per-expert bias added to scores
hidden_states        [T, 7168]       FP8     the actual token data (compressed)
hidden_states_scale  [56, T]         FP32    dequant scales (TRANSPOSED!)
gemm1_weights        [32, 4096, 7168] FP8    expert W1‖W3 matrices (compressed)
gemm1_weights_scale  [32, 32, 56]    FP32    block scales for W13
gemm2_weights        [32, 7168, 2048] FP8    expert W2 matrices (compressed)
gemm2_weights_scale  [32, 56, 16]    FP32    block scales for W2
local_expert_offset  scalar          INT     global ID of local expert 0
routed_scaling_factor scalar         FP32    final scaling constant
output               [T, 7168]       BF16    PRE-ALLOCATED output (DPS)
```

**Key numbers:**
- H = 7168 (hidden size)
- I = 2048 (intermediate size)
- 2I = 4096 (GEMM1 output has gate and up projections concatenated)
- 56 = 7168 / 128 (number of 128-element blocks in hidden dim)
- 32 = 4096 / 128 (number of 128-element blocks in 2I dim)
- 16 = 2048 / 128 (number of 128-element blocks in I dim)

---

## Stage 1: FP8 Dequantization (Lines 288-292)

### What happens
Every element in `hidden_states` is a tiny 8-bit float. To do real math, we multiply
each by a scale factor to get a proper FP32 value.

### The math
```
a[t, h] = hidden_states[t, h]  ×  hidden_states_scale[h ÷ 128, t]
         ─────────────────────     ──────────────────────────────────
          FP8 value (±448 max)      block scale (one per 128 elements)
```

### Why `[56, T]` not `[T, 56]`?
The scale tensor is **transposed**. Row = which 128-block of hidden dim. Column = which token.
This is a data layout choice from the model — we can't change it.

### What the Triton kernel does (lines 41-65)

```
Grid: (⌈T/32⌉, ⌈7168/128⌉) = (⌈T/32⌉, 56) thread blocks

Each thread block handles a 32×128 tile of the output:
  1. Compute which rows (tokens 0..31) and cols (hidden 0..127) this block owns
  2. Load 32×128 of FP8 values → cast to FP32
  3. For each col, compute h_block = col ÷ 128  (which scale group)
  4. Load 32×128 scale values from scale[h_block, token]
  5. Multiply: output = fp8_value × scale
  6. Store 32×128 FP32 result

Memory per block: 32×128×4 bytes × 3 (input + scale + output) ≈ 48 KB
```

### Profiling estimate
```
Data moved:    T × 7168 × 1B (FP8 read) + T × 56 × 4B (scale read) + T × 7168 × 4B (FP32 write)
             = T × (7168 + 224 + 28672) = T × 36064 bytes
For T=1024:  ~35 MB → at 8 TB/s = 0.004 ms (memory-bound, nearly free)
```

---

## Stage 2: Routing (Lines 294-320)

This is the brain that decides which experts process which tokens.
All done in PyTorch (not Triton) — ~10 GPU kernel launches.

### Step-by-step operations

**Step 2a: Sigmoid scores** (line 298)
```
s[t, e] = σ(routing_logits[t, e]) = 1 / (1 + exp(-logits[t, e]))
```
Converts raw logits to probabilities in (0, 1). Shape: `[T, 256]`

**Step 2b: Add bias** (line 299)
```
s_with_bias[t, e] = s[t, e] + bias[e]
```
Bias nudges some experts to be picked more often. Shape: `[T, 256]`

**Step 2c: Group scoring** (lines 301-303)
```
Reshape [T, 256] → [T, 8, 32]     (8 groups of 32 experts)
For each group g:
    group_score[t, g] = sum of top-2 values in s_with_bias[t, g, :]
```
Each group gets a score = sum of its 2 best experts. Shape: `[T, 8]`

**Step 2d: Select top-4 groups** (lines 305-308)
```
group_idx[t, :] = indices of top-4 groups by group_score
group_mask[t, g] = True if group g is in top-4
```
We keep 4 out of 8 groups → 4×32 = 128 candidate experts remain.

**Step 2e: Build expert mask** (lines 310-313)
```
score_mask[t, e] = True if expert e's group is in the top-4
scores_pruned[t, e] = s_with_bias[t, e]  if mask is True
                    = -∞                  otherwise
```
Kills all experts from non-selected groups. Shape: `[T, 256]`

**Step 2f: Select top-8 experts** (lines 315-316)
```
topk_idx[t, k] = expert index of the k-th best expert (k=0..7)
```
From the 128 surviving experts, pick the best 8. Shape: `[T, 8]`

**Step 2g: Compute weights** (lines 318-320)
```
topk_s[t, k]  = s[t, topk_idx[t,k]]         ← use UNBIASED sigmoid!
topk_w[t, k]  = topk_s[t,k] / sum_k(topk_s[t,:])   ← normalize
topk_w[t, k] *= routed_scaling_factor        ← global scale
```
**Critical:** Weights use `s` (no bias), not `s_with_bias`. Shape: `[T, 8]`

### Profiling estimate
```
~10 PyTorch kernel launches
For T=1024: topk on [1024, 256] is small → total ~0.3-0.5 ms
For T=1:    still ~0.3 ms (launch overhead dominates)
```

### Flowchart
```
logits [T,256]
   │
   ├──→ sigmoid ──→ s [T,256] ──────────────────────────┐
   │                    │                                │
   │              + bias[256]                    (used for weights)
   │                    │                                │
   │              s_with_bias [T,256]                    │
   │                    │                                │
   │              view [T,8,32]                          │
   │                    │                                │
   │              top2 per group → sum                   │
   │                    │                                │
   │              group_scores [T,8]                     │
   │                    │                                │
   │              top4 groups                            │
   │                    │                                │
   │              mask out losers                        │
   │                    │                                │
   │              topk(k=8) ──→ topk_idx [T,8]          │
   │                                                    │
   │                           gather s at topk_idx ←───┘
   │                                   │
   │                           normalize → topk_w [T,8]
   │                                   │
   │                              × scale_factor
   │                                   │
   └───────────────────────────────────┘
```

---

## Stage 3: Dispatch Table (Lines 322-351)

### What happens
We need to know: for each expert, which tokens go to it?

### The naive way (sub-1/sub-2)
```python
for expert in range(32):
    mask = (local_idx == expert) & valid    # GPU kernel
    any_match = mask.any()                   # GPU kernel
    token_ids = mask.nonzero()               # GPU kernel
    # = 96 GPU kernel launches for 32 experts
```

### The optimized way (sub-3+)
```python
# Step 1: Find all valid (token, topk_position) pairs → 1 kernel
all_valid = nonzero(valid_local)              # [N_valid, 2]

# Step 2: Get their expert IDs → indexing (free)
expert_ids = local_idx[all_valid[:,0], all_valid[:,1]]

# Step 3: Sort by expert → 1 kernel
sorted_order = argsort(expert_ids)

# Step 4: Find where each expert's tokens start/end → 1 kernel
unique_experts, counts = unique_consecutive(sorted_expert_ids)
# → then split into per-expert lists

# Total: ~4 GPU kernel launches instead of ~96
```

### Example
```
Suppose T=4 tokens, topk=2 (simplified), 3 local experts:

topk_idx = [[0, 2],   ← token 0 goes to experts 0, 2
             [1, 0],   ← token 1 goes to experts 1, 0
             [2, 1],   ← token 2 goes to experts 2, 1
             [0, 1]]   ← token 3 goes to experts 0, 1

After dispatch:
  Expert 0: tokens [0, 1, 3], topk_pos [0, 1, 0]
  Expert 1: tokens [1, 2, 3], topk_pos [1, 1, 1]
  Expert 2: tokens [0, 2],    topk_pos [1, 0]
```

---

## Stage 4: Expert Compute (Lines 353-393)

This is the expensive part. For each of 32 experts, process its tokens.

### For each expert `le`:

#### Step 4a: Gather tokens (line 363)
```
a_e = a[token_idx, :]     shape: [Tk, 7168]
```
Pick rows of the dequanted hidden states for tokens assigned to this expert.

#### Step 4b: GEMM1 — Up+Gate projection (lines 366-373)

**The math:**
```
g1 = a_e × W13ᵀ

Where:
  a_e :  [Tk, 7168]  FP32    (dequanted activations)
  W13 :  [4096, 7168] FP8    (concatenated up + gate weights)
  g1 :   [Tk, 4096]  FP32    (result)
```

**Two paths:**

*Path A — Fused (Tk ≥ 16):* Triton kernel does GEMM + weight dequant simultaneously
```
For each 128-element K-chunk:
  1. Load A tile [BLOCK_M, 128] from activations
  2. Load W tile [BLOCK_N, 128] from FP8 weights
  3. Load 1 scale per N-row: S[n÷128, k÷128]
  4. Dequant: W_fp32 = W_fp8 × scale
  5. Multiply: acc += A_tile × W_fp32ᵀ
Never materializes full 4096×7168 FP32 weight = saves 112 MB per expert
```

*Path B — Fallback (Tk < 16):* Pre-dequant weight then cuBLAS matmul
```
1. Cast W13 to FP32: view as [32, 128, 56, 128] blocks
2. Multiply each block by its scale: W_fp32 = W_fp8 × S
3. Reshape back to [4096, 7168]
4. Standard matmul: g1 = a_e × W13_fp32ᵀ
```

#### Step 4c: SwiGLU activation (lines 376-379)

**The math:**
```
Split g1 into two halves:
  x_up   = g1[:, 0:2048]      (up projection)
  x_gate = g1[:, 2048:4096]   (gate projection)

c = x_up × silu(x_gate)
  = x_up × (x_gate × σ(x_gate))
  = x_up × (x_gate / (1 + exp(-x_gate)))

Output c: [Tk, 2048]
```

**What silu does visually:**
```
silu(x):  negative x → ~0 (gate closes)
          x = 0     → 0
          positive x → ~x (gate opens, linear)

It's a smooth "on/off switch" that lets the gate control
which dimensions of the up projection survive.
```

**Triton kernel (lines 68-90):**
```
Each thread block processes a [64, 128] tile:
  1. Load x_up  from columns [0:I]
  2. Load x_gate from columns [I:2I]
  3. Compute: x_up × x_gate × sigmoid(x_gate)
  4. Store to output

All in FP32 registers — never writes intermediate sigmoid to memory.
```

#### Step 4d: GEMM2 — Down projection (lines 382-389)

**The math:**
```
o = c × W2ᵀ

Where:
  c :   [Tk, 2048]  FP32
  W2 :  [7168, 2048] FP8
  o :   [Tk, 7168]  FP32

Same two-path logic as GEMM1 (fused or fallback).
```

#### Step 4e: Weighted accumulation (lines 392-393)

**The math:**
```
For each token t assigned to this expert at topk position k:
  accum[t, :] += o[t, :] × topk_w[t, k]
```

`index_add_` is an atomic scatter-add: multiple experts writing to the same
token's row will correctly sum up. Each expert contributes its weighted share.

### Profiling estimate per expert (Tk=32 tokens)
```
GEMM1: [32, 7168] × [7168, 4096] = 32 × 7168 × 4096 × 2 = 1.88 GFLOP
SwiGLU: 32 × 2048 × 3 ops = 0.2 MFLOP  (negligible)
GEMM2: [32, 2048] × [2048, 7168] = 32 × 2048 × 7168 × 2 = 0.94 GFLOP
Total per expert: ~2.82 GFLOP

At B200's ~4000 TFLOPS FP8 (or ~60 TFLOPS FP32):
  FP32 path: 2.82 / 60 = 0.047 ms per expert
  × 32 experts = 1.5 ms  (compute only — add data movement)

Memory per expert (fused path):
  Read W13 FP8: 4096 × 7168 × 1B = 28 MB
  Read W2 FP8:  7168 × 2048 × 1B = 14 MB
  Total: 42 MB at 8 TB/s = 0.005 ms  (bandwidth is not bottleneck)
```

---

## Stage 5: Output (Lines 395-396)

```
output.copy_(accum.to(torch.bfloat16))
```

1. Cast `accum [T, 7168]` from FP32 to BF16
2. Copy into the pre-allocated `output` buffer (DPS)

This is a single memcpy — negligible cost.

---

## Complete Math in One Equation

For each token `t`, the output is:

```
                  8
output[t] = Σ   w_k × W2_e(k)ᵀ · SwiGLU( W13_e(k)ᵀ · dequant(x[t]) )
                 k=1

Where:
  e(k)     = topk_idx[t, k]              (selected expert)
  w_k      = topk_w[t, k]                (normalized routing weight)
  x[t]     = hidden_states[t, :]          (FP8 input)
  dequant  = x_fp8 × scale[h÷128, t]     (block-scale FP8→FP32)
  SwiGLU(y)= y[:I] × silu(y[I:])         (gated activation)
  silu(z)  = z × σ(z)                    (smooth gating)
  W13, W2  = expert weight matrices       (FP8, block-dequanted on-the-fly)
```

---

## Profiling: Where Time Goes

### Measured timing breakdown (from sub-3, B200)

Using the 3 benchmark tiers as proxy:

**Small-T workloads (T ≈ 1-8):** Total ~1.7-3.2 ms
```
┌─────────────────────────────────────────────────────────┐
│ Routing (PyTorch ops) ██████████████████████████   ~60% │
│ Dispatch table        ████                         ~10% │
│ Expert compute        ████████                     ~20% │
│ Dequant + output      ███                          ~10% │
└─────────────────────────────────────────────────────────┘
Bottleneck: KERNEL LAUNCH OVERHEAD (routing does ~10 torch ops)
```

**Medium-T workloads (T ≈ 64-256):** Total ~5-10 ms
```
┌─────────────────────────────────────────────────────────┐
│ Routing               ████                         ~10% │
│ Dispatch table        ██                            ~5% │
│ Expert compute        ██████████████████████████   ~75% │
│ Dequant + output      ███                          ~10% │
└─────────────────────────────────────────────────────────┘
Bottleneck: SEQUENTIAL EXPERT LOOP (32 experts × GEMM1+SwiGLU+GEMM2)
```

**Large-T workloads (T ≈ 512-2048):** Total ~10-25 ms
```
┌─────────────────────────────────────────────────────────┐
│ Routing               ██                            ~5% │
│ Dispatch table        █                             ~3% │
│ Expert compute        ████████████████████████████  ~85% │
│ Dequant + output      ██                            ~7% │
└─────────────────────────────────────────────────────────┘
Bottleneck: WEIGHT LOADING + GEMM COMPUTE (each expert loads ~42 MB of weights)
```

### B200 hardware utilization estimate

```
For the largest workload (5e8dc11c, 24.4 ms):

Total weight data per forward:
  W13: 32 experts × 4096 × 7168 × 1B = 896 MB
  W2:  32 experts × 7168 × 2048 × 1B = 448 MB
  Total: 1.34 GB

  At 8 TB/s → minimum read time = 0.17 ms
  Actual time = 24.4 ms → we're using 0.7% of peak bandwidth!

  Why so slow? Sequential expert loop means:
  - 32 × kernel launch overhead
  - 32 × separate weight reads (no reuse)
  - 32 × separate matmuls (no batching)
  - Python loop overhead between experts
```

### Theoretical minimum time

```
Compute:
  GEMM1: T×8 × 7168 × 4096 × 2 FLOP    (each token hits ~8 experts)
  GEMM2: T×8 × 2048 × 7168 × 2 FLOP
  Total: T×8 × 2 × (7168×4096 + 2048×7168)  FLOP

  For T=512: 8 × 2 × 512 × (29M + 14.7M) = 358 GFLOP
  At 60 TFLOPS FP32: 358/60000 = 6.0 ms
  At 4000 TFLOPS FP8: 358/4000000 = 0.09 ms (if we used FP8 tensor cores!)

Memory:
  Weight reads: 1.34 GB (if reading each expert once)
  At 8 TB/s: 0.17 ms

Theoretical minimum: max(compute, memory) ≈ 0.2 ms (with FP8 tensor cores)
                     vs actual 24.4 ms → 120× gap!
```

### Where the gap comes from

```
1. FP32 GEMM instead of FP8 tensor cores     → 60× slower compute
2. Sequential expert loop                      → 32× less parallelism  
3. Python loop overhead                        → ~1 ms wasted
4. Separate kernel launches per expert         → 0.05 ms × 32 = 1.6 ms
5. No weight reuse across tokens               → redundant memory traffic
```

---

## What Each Line Does (Quick Reference)

| Lines | Operation | Shape In → Out | GPU Work |
|-------|-----------|---------------|----------|
| 288-289 | Ensure contiguous memory | — | memcpy if needed |
| 292 | FP8 dequant hidden | `[T,H] fp8 → [T,H] fp32` | Triton kernel |
| 295-296 | Cast logits/bias to FP32 | `[T,256],[256] → fp32` | PyTorch |
| 298 | Sigmoid | `[T,256] → [T,256]` | PyTorch |
| 299 | Add bias | `[T,256]+[256] → [T,256]` | PyTorch |
| 301 | Reshape | `[T,256] → [T,8,32]` | Free (view) |
| 302-303 | Top-2 per group + sum | `[T,8,32] → [T,8]` | PyTorch |
| 305-308 | Top-4 groups + mask | `[T,8] → [T,8] bool` | PyTorch |
| 310-313 | Expand mask + prune | `[T,8] → [T,256]` | PyTorch |
| 315-316 | Top-8 experts | `[T,256] → [T,8]` | PyTorch |
| 318-320 | Normalize weights | `[T,8] → [T,8]` | PyTorch |
| 323-324 | Map to local experts | `[T,8] → [T,8]` | PyTorch |
| 330-344 | Build dispatch table | `[T,8] → 32 lists` | PyTorch |
| 354 | Zero accumulator | `[T,H]` | PyTorch |
| 363 | Gather tokens | `[T,H] → [Tk,H]` | PyTorch |
| 367-370 | Fused GEMM1+dequant | `[Tk,H]×[2I,H] → [Tk,2I]` | Triton |
| 372-373 | Fallback GEMM1 | same | cuBLAS |
| 377 | SwiGLU (Triton) | `[Tk,2I] → [Tk,I]` | Triton |
| 379 | SwiGLU (PyTorch) | same | PyTorch |
| 383-386 | Fused GEMM2+dequant | `[Tk,I]×[H,I] → [Tk,H]` | Triton |
| 388-389 | Fallback GEMM2 | same | cuBLAS |
| 392-393 | Weighted scatter-add | `[Tk,H] → accum[T,H]` | PyTorch |
| 396 | Cast + copy to output | `[T,H] fp32 → bf16` | PyTorch |

---

## Data Flow Through Memory (What Blackwell Sees)

```
HBM3e (192 GB, 8 TB/s)
 ├── hidden_states [T, 7168] FP8         ← read once
 ├── hidden_states_scale [56, T] FP32    ← read once
 ├── routing_logits [T, 256] FP32        ← read once
 ├── routing_bias [256] BF16             ← read once (tiny)
 ├── gemm1_weights [32, 4096, 7168] FP8  ← read 32× (once per expert)
 ├── gemm1_weights_scale [32, 32, 56]    ← read 32× (tiny per expert)
 ├── gemm2_weights [32, 7168, 2048] FP8  ← read 32× (once per expert)
 ├── gemm2_weights_scale [32, 56, 16]    ← read 32× (tiny per expert)
 └── output [T, 7168] BF16              ← write once

L2 Cache (126 MB)
 ├── Expert weights: 28+14 = 42 MB per expert (FITS in L2!)
 ├── Activations for one tile
 └── Routing intermediates

SMEM (228 KB per SM, 148 SMs = 33.7 MB total)
 ├── GEMM tiles: BLOCK_M × BLOCK_K × 4B + BLOCK_N × BLOCK_K × 4B
 │   e.g., 128×128×4 + 128×128×4 = 128 KB (fits in 228 KB SMEM)
 └── Scale values: BLOCK_N × 4B = 512B (negligible)

Registers (256 KB per SM)
 └── Accumulator: BLOCK_M × BLOCK_N × 4B = 64 KB (register-held)
```

---

> **Last updated:** 2026-03-12 | Based on kernel.py submission-4
