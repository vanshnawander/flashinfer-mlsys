# Triton Kernel Concepts — Fused MoE Implementation Guide

> Complete reference for every Triton function, parameter, and concept used in the
> `kernel.py` implementation for the FlashInfer MLSys'26 fused MoE track.
> Target hardware: **NVIDIA B200 (Blackwell)** — 148 SMs, 228 KB shared memory/SM, 8 TB/s HBM3e.

---

## Table of Contents

1. [Kernel Overview & Architecture](#1-kernel-overview--architecture)
2. [Destination Passing Style (DPS)](#2-destination-passing-style-dps)
3. [Triton Core Functions Reference](#3-triton-core-functions-reference)
4. [Kernel 1: FP8 Hidden State Dequantization](#4-kernel-1-fp8-hidden-state-dequantization)
5. [Kernel 2: SwiGLU Activation](#5-kernel-2-swiglu-activation)
6. [Stage 2: DeepSeek-V3 No-Aux Routing](#6-stage-2-deepseek-v3-no-aux-routing)
7. [Stage 3: Expert Compute Pipeline](#7-stage-3-expert-compute-pipeline)
8. [FP8 Block-Scale Dequantization Deep Dive](#8-fp8-block-scale-dequantization-deep-dive)
9. [B200 Blackwell Optimization Notes](#9-b200-blackwell-optimization-notes)
10. [Common Pitfalls & Debugging](#10-common-pitfalls--debugging)
11. [CUDA Kernel Equivalents](#11-cuda-kernel-equivalents)
12. [References & Citations](#12-references--citations)

---

## 1. Kernel Overview & Architecture

The fused MoE kernel implements one layer of a **DeepSeek-V3 style Mixture-of-Experts** network. The pipeline is:

```
FP8 Hidden States ──→ [Dequant] ──→ FP32 Activations
                                         │
Routing Logits ──→ [Routing] ──→ topk_idx, topk_w
                                         │
                              ┌──────────┘
                              ▼
                    For each local expert:
                    ┌─────────────────────┐
                    │ GEMM1: A×W13ᵀ      │ → [Tk, 2I]
                    │ SwiGLU activation   │ → [Tk, I]
                    │ GEMM2: C×W2ᵀ       │ → [Tk, H]
                    │ Weighted accumulate │
                    └─────────────────────┘
                              │
                              ▼
                    output [T, H] bfloat16
```

### Fixed Geometry (from definition name)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `H` | 7168 | Hidden dimension |
| `I` | 2048 | Intermediate (FFN) dimension |
| `E_global` | 256 | Total expert count |
| `E_local` | 32 | Experts on this rank |
| `TOP_K` | 8 | Experts selected per token |
| `N_GROUP` | 8 | Routing groups |
| `TOPK_GROUP` | 4 | Groups kept per token |
| `BLOCK_Q` | 128 | FP8 quantization block size |

### Kernel Inputs

| Name | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `routing_logits` | `[T, 256]` | `float32` | Pre-sigmoid routing scores |
| `routing_bias` | `[256]` | `bfloat16` | Expert bias for routing |
| `hidden_states` | `[T, 7168]` | `float8_e4m3fn` | FP8-quantized activations |
| `hidden_states_scale` | `[56, T]` | `float32` | Per-block scales (**transposed!**) |
| `gemm1_weights` | `[32, 4096, 7168]` | `float8_e4m3fn` | W1‖W3 concatenated |
| `gemm1_weights_scale` | `[32, 32, 56]` | `float32` | Block scales for W13 |
| `gemm2_weights` | `[32, 7168, 2048]` | `float8_e4m3fn` | W2 (down projection) |
| `gemm2_weights_scale` | `[32, 56, 16]` | `float32` | Block scales for W2 |
| `local_expert_offset` | scalar | `int32` | Global ID of local expert 0 |
| `routed_scaling_factor` | scalar | `float32` | Scaling factor for weights |

### Kernel Output (DPS)

| Name | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `output` | `[T, 7168]` | `bfloat16` | Pre-allocated, written in-place |

---

## 2. Destination Passing Style (DPS)

**Critical concept that caused the original kernel failure.**

FlashInfer-Bench defaults to `destination_passing_style: true` in the `BuildSpec`. This means:

```python
# DPS=true: framework calls kernel like this
kernel(input1, input2, ..., inputN, output)
#                                    ^^^^^^ pre-allocated output tensor

# DPS=false: framework expects return value
result = kernel(input1, input2, ..., inputN)
```

**Why DPS matters:**
- Avoids tensor allocation overhead in benchmarks → more accurate timing
- The output tensor is pre-allocated by the framework with the correct shape and dtype
- The kernel must write results in-place using `output.copy_()` or slice assignment

**The original bug:** The kernel signature did not include an `output` parameter, causing an
argument count mismatch when the framework tried to call it with the extra output tensor.

**Fix:** Added `output: torch.Tensor` as the last parameter and use `output.copy_(result)`.

> **Reference:** [FlashInfer-Bench Solution Schema](https://bench.flashinfer.ai/docs/flashinfer-trace/solution) [1]

---

## 3. Triton Core Functions Reference

### 3.1 `@triton.jit`

**Purpose:** JIT-compiles a Python function into a GPU kernel.

```python
@triton.jit
def my_kernel(x_ptr, y_ptr, N: tl.constexpr):
    ...
```

- Functions decorated with `@triton.jit` are compiled to PTX/CUBIN at first call
- Arguments can be pointers (tensor data), scalars, or `tl.constexpr` (compile-time constants)
- The function body uses `triton.language` (tl) operations — NOT regular Python

> **Reference:** [Triton Documentation](https://triton-lang.org/main/python-api/triton.html) [2]

### 3.2 `tl.program_id(axis)`

**Purpose:** Returns the index of the current program instance (thread block) along a given axis.

```python
pid_m = tl.program_id(0)   # block index along axis 0
pid_n = tl.program_id(1)   # block index along axis 1
```

- **Input:** `axis` — integer 0, 1, or 2 (matching the launch grid dimensions)
- **Output:** scalar integer — the block ID along that axis
- Analogous to CUDA's `blockIdx.x`, `blockIdx.y`, `blockIdx.z`
- The grid dimensions are set when launching: `kernel[grid](...)`

**In our kernels:**
- `pid_m` → indexes along the token (row) dimension
- `pid_n` → indexes along the hidden (column) dimension

> **Reference:** [Triton Language — program_id](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html) [3]

### 3.3 `tl.arange(start, end)`

**Purpose:** Creates a 1D tensor of contiguous integers `[start, start+1, ..., end-1]`.

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
# e.g., if pid_m=2, BLOCK_M=32: [64, 65, 66, ..., 95]
```

- **Input:** `start`, `end` — integers; `end - start` must be a power of 2
- **Output:** 1D tensor of `int32` values
- Used to compute per-thread element offsets within a tile
- Combined with `program_id` to get global indices

**Constraint:** `end - start` ≤ 1,048,576 (TRITON_MAX_TENSOR_NUMEL)

> **Reference:** [Triton Language — arange](https://triton-lang.org/main/python-api/generated/triton.language.arange.html) [4]

### 3.4 `tl.load(pointer, mask=None, other=0.0)`

**Purpose:** Loads data from global memory into registers.

```python
x_ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
x = tl.load(x_ptrs, mask=mask, other=0.0)
```

- **Input:**
  - `pointer` — a tensor of memory addresses (computed from base pointer + offsets)
  - `mask` — boolean tensor; only load where mask is `True`
  - `other` — value to use for masked-out elements (default: 0.0)
- **Output:** tensor of loaded values (same shape as pointer tensor)
- Masked loads prevent out-of-bounds memory access (critical for edge tiles)

**Cache modifiers (optional):**
- `cache_modifier="ca"` — cache at all levels (default)
- `cache_modifier=".cg"` — cache at global level only
- `eviction_policy=".evict_first"` — hint for cache eviction

> **Reference:** [Triton Language — load](https://triton-lang.org/main/python-api/generated/triton.language.load.html) [5]

### 3.5 `tl.store(pointer, value, mask=None)`

**Purpose:** Stores data from registers to global memory.

```python
o_ptrs = o_ptr + offs_m[:, None] * so0 + offs_n[None, :] * so1
tl.store(o_ptrs, out, mask=mask)
```

- **Input:**
  - `pointer` — tensor of target memory addresses
  - `value` — tensor of values to store (must match pointer shape)
  - `mask` — boolean tensor; only store where mask is `True`
- No return value
- Masked stores prevent writing out-of-bounds

> **Reference:** [Triton Language — store](https://triton-lang.org/main/python-api/generated/triton.language.store.html) [6]

### 3.6 `tl.sigmoid(x)`

**Purpose:** Computes element-wise sigmoid: `σ(x) = 1 / (1 + exp(-x))`

```python
silu_x2 = x2 * tl.sigmoid(x2)   # silu(x) = x * sigmoid(x)
```

- **Input:** tensor of float values
- **Output:** tensor of float values in range (0, 1)
- Used in our SwiGLU kernel to compute `silu(x) = x * σ(x)`
- Numerically stable implementation built into Triton

> **Reference:** [Triton Language — sigmoid](https://triton-lang.org/main/python-api/generated/triton.language.sigmoid.html) [7]

### 3.7 `.to(dtype)` (Type Casting)

**Purpose:** Casts a tensor to a different data type.

```python
x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
```

- Used extensively for FP8 → FP32 conversion
- Supported types: `tl.float16`, `tl.float32`, `tl.bfloat16`, `tl.int32`, etc.
- **Critical for FP8:** Always cast to FP32 before arithmetic to avoid precision loss

### 3.8 `tl.constexpr`

**Purpose:** Marks a kernel parameter as a compile-time constant.

```python
def _kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
```

- The compiler can use these values for loop unrolling, register allocation, etc.
- Different `constexpr` values create different compiled kernel variants
- Tile sizes (`BLOCK_M`, `BLOCK_N`) should always be `tl.constexpr`

### 3.9 `triton.cdiv(a, b)`

**Purpose:** Ceiling division — `⌈a/b⌉`

```python
grid = (triton.cdiv(t_size, BLOCK_M), triton.cdiv(h_size, BLOCK_N))
```

- Used to compute grid dimensions to cover the full tensor
- Ensures all elements are processed even when dimensions aren't divisible by block size

### 3.10 Kernel Launch: `kernel[grid](...)`

**Purpose:** Launches the Triton kernel on the GPU with the specified grid.

```python
_dequant_hidden_fp8_kernel[grid](
    hidden_states, hidden_states_scale, out,
    t_size, h_size,
    hidden_states.stride(0), hidden_states.stride(1),
    ...
    BLOCK_M=32, BLOCK_N=128, SCALE_BLOCK=128,
)
```

- `grid` is a tuple of (grid_x, grid_y, grid_z) — number of blocks per axis
- Arguments are passed positionally to the `@triton.jit` function
- `stride(dim)` gives the number of elements between consecutive entries along `dim`
- `constexpr` arguments are passed as keyword arguments

---

## 4. Kernel 1: FP8 Hidden State Dequantization

### What it does

Converts FP8-quantized hidden states to FP32 using block-wise scales:

```
output[t, h] = (float)hidden_states[t, h] × scale[h ÷ 128, t]
```

### Scale layout: `[H/128, T]` — TRANSPOSED

This is a critical subtlety. The scale tensor has shape `[H/128, T]` where:
- **Row** = which 128-element block of the hidden dimension
- **Column** = which token

This is **transposed** compared to the intuitive `[T, H/128]` layout. The Triton kernel
accounts for this by computing pointer arithmetic accordingly:

```python
h_block = offs_n // SCALE_BLOCK   # which 128-group of hidden dim
s_ptrs = s_ptr + h_block[None, :] * ss0 + offs_m[:, None] * ss1
#                ^^^^^^ row dim (h_block)   ^^^^^^ col dim (token)
```

### Tiling Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BLOCK_M` | 32 | Tokens per tile — small to handle varying `T` |
| `BLOCK_N` | 128 | Hidden per tile — matches BLOCK_Q for aligned scale access |

Grid: `(⌈T/32⌉, ⌈H/128⌉)` → e.g., for T=1024: `(32, 56)` = 1,792 blocks

### Memory Access Pattern

- **Hidden states:** Coalesced reads — threads within a warp read consecutive hidden elements
- **Scales:** Broadcast pattern — all threads in a column read the same scale value for their h_block
- **Output:** Coalesced writes — same pattern as hidden state reads

---

## 5. Kernel 2: SwiGLU Activation

### What is SwiGLU?

SwiGLU is a gated activation function used in modern LLMs (LLaMA, DeepSeek, PaLM):

```
SwiGLU(X1, X2) = X1 × silu(X2)
where silu(x) = x × σ(x) = x × sigmoid(x)
```

### Input/Output

- **Input:** `g1 [Tk, 2×I]` — concatenated output of GEMM1
  - First `I` columns: X1 (up projection)
  - Last `I` columns: X2 (gate projection)
- **Output:** `c [Tk, I]` — the activated values

### How it works in Triton

```python
# Two loads from the same input row at different column offsets
x1_ptrs = g1_ptr + offs_m[:, None] * sg10 + offs_n[None, :] * sg11           # [:, 0:I]
x2_ptrs = g1_ptr + offs_m[:, None] * sg10 + (offs_n[None, :] + i_size) * sg11  # [:, I:2I]

x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

silu_x2 = x2 * tl.sigmoid(x2)   # silu activation on gate
c = x1 * silu_x2                # element-wise gating
```

### Why SwiGLU in Triton?

- Avoids materializing intermediate `silu` result in global memory
- Fused load + compute + store in one kernel launch
- FP32 precision throughout prevents numerical drift

> **Reference:** Shazeer, "GLU Variants Improve Transformer", 2020 [8]

---

## 6. Stage 2: DeepSeek-V3 No-Aux Routing

This routing is implemented in **PyTorch** (not Triton) for correctness. Here's the algorithm:

### Step-by-step

```
1. s = sigmoid(routing_logits)                    # [T, 256]
2. s_with_bias = s + routing_bias                 # [T, 256]
3. Group into 8 groups of 32:
   s_grouped = s_with_bias.view(T, 8, 32)        # [T, 8, 32]
4. Per group: sum of top-2 values → group score
   group_scores = topk(s_grouped, k=2).sum()      # [T, 8]
5. Keep top-4 groups
   group_mask = topk(group_scores, k=4)            # [T, 8] bool
6. Within kept groups, pick global top-8 experts
   topk_idx = topk(masked_scores, k=8)             # [T, 8] int
7. Normalize weights using s (WITHOUT bias):
   topk_w = s[topk_idx] / sum(s[topk_idx]) × scale_factor
```

### Critical Detail: Bias vs No-Bias

- **Routing uses `s_with_bias`** for expert selection (steps 2–6)
- **Weights use `s` (without bias)** for normalization (step 7)
- Using biased scores for normalization is a **common pitfall** that causes correctness failures

> **Reference:** DeepSeek-V3 Technical Report, Section 3.3 [9]

---

## 7. Stage 3: Expert Compute Pipeline

### Per-Expert Loop

```python
for le in range(NUM_LOCAL_EXPERTS):   # 0..31
    # Find tokens routed to this expert
    sel = valid_local & (local_idx == le)
    token_idx, topk_pos = torch.nonzero(sel, as_tuple=True)

    # Gather activations for selected tokens
    a_e = a[token_idx]                           # [Tk, H]

    # GEMM1: activation × W13ᵀ
    g1 = a_e @ w13_e.t()                         # [Tk, 2I]

    # SwiGLU activation (Triton kernel)
    c = _swiglu(g1)                              # [Tk, I]

    # GEMM2: activated × W2ᵀ
    o = c @ w2_e.t()                             # [Tk, H]

    # Weighted scatter-add to output
    output[token_idx] += o × topk_w[token_idx, topk_pos]
```

### Weight Dequantization

Each expert's weights are dequantized per-block:

```python
# Shape: [2I, H] → view as [2I/128, 128, H/128, 128]
w = w13_e.view(n_out_blocks, 128, n_h_blocks, 128)
s = s13_e.view(n_out_blocks,   1, n_h_blocks,   1)
result = (w * s).reshape(2*I, H)   # broadcast scale over 128×128 blocks
```

This is equivalent to the reference's `repeat_interleave` but more memory-efficient.

---

## 8. FP8 Block-Scale Dequantization Deep Dive

### What is FP8 (float8_e4m3fn)?

| Field | Bits | Description |
|-------|------|-------------|
| Sign | 1 | Sign bit |
| Exponent | 4 | Biased exponent (bias=7) |
| Mantissa | 3 | Explicit mantissa bits |

- **Range:** ±448 (max representable value)
- **Precision:** ~3.5 decimal digits
- The "fn" suffix means "finite" — no infinity representation
- NaN: bit patterns `0x7F` and `0xFF`

### Why Block Scaling?

Per-tensor scaling (one scale for entire tensor) loses too much precision when values
have varying magnitudes. **Block scaling** assigns a separate FP32 scale to each
128-element block, enabling:

- Better dynamic range utilization per block
- 4× memory savings over FP32 with minimal accuracy loss
- Compatible with hardware tensor core FP8 operations

### Scale Index Formulas

```
Hidden scale:    scale[h ÷ 128, token_index]
GEMM1 scale:     scale[expert, out_col ÷ 128, h ÷ 128]
GEMM2 scale:     scale[expert, h ÷ 128, intermediate ÷ 128]
```

> **Reference:** NVIDIA FP8 Formats, CUDA Math API Documentation [10]

---

## 9. B200 Blackwell Optimization Notes

### Key Hardware Specs

| Feature | B200 Value | Impact on MoE |
|---------|-----------|---------------|
| SMs | 148 | More parallelism for expert loop |
| Shared Memory/SM | 228 KB | Larger tiles, cache scales in SMEM |
| HBM3e Bandwidth | 8 TB/s | Faster weight loads |
| FP8 TFLOPS | ~4,000 | On-the-fly dequant in tensor cores |
| FP4 TFLOPS | ~8,000 | Future optimization path |
| L2 Cache | 126 MB | Can cache routing data |
| Max Warps/SM | 64 | Higher occupancy potential |

### Optimization Strategies for B200

1. **Larger BLOCK_N tiles (256 or 512):**
   - B200's 228 KB SMEM can hold bigger tiles
   - Reduces global memory traffic for weight dequant

2. **FP8 on-the-fly dequant in GEMMs:**
   - Instead of materializing full dequantized weights, load FP8 tiles and apply scales in SMEM
   - Saves ~14× memory for W13 (32 experts × 4096×7168×1B vs ×4B)

3. **Token reordering by expert:**
   - Sort tokens by expert assignment before compute
   - Enables contiguous memory access in the expert loop

4. **Persistent kernel for expert loop:**
   - Run all 32 experts in one kernel launch
   - Avoid launch overhead per expert

5. **Use `tl.dot()` for GEMMs:**
   - Triton's `tl.dot()` maps directly to tensor core MMA instructions
   - Required for achieving peak FP8/FP16 throughput

> **Reference:** NVIDIA B200 Whitepaper, Blackwell Architecture Guide [11]

---

## 10. Common Pitfalls & Debugging

### Pitfall 1: DPS Signature Mismatch ⚠️

**Symptom:** Kernel fails immediately with argument count error
**Cause:** `destination_passing_style=true` but kernel doesn't accept output parameter
**Fix:** Add `output: torch.Tensor` as last parameter, write in-place

### Pitfall 2: Using Biased Scores for Normalization

**Symptom:** Incorrect routing weights, wrong expert contributions
**Cause:** Normalizing with `s_with_bias` instead of `s` (without bias)
**Fix:** Gather from `s` (raw sigmoid) for normalization: `topk_s = s.gather(1, topk_idx)`

### Pitfall 3: Wrong Scale Layout

**Symptom:** Garbage output values, nans
**Cause:** `hidden_states_scale` is `[H/128, T]` not `[T, H/128]`
**Fix:** Index as `scale[h_block, token]` not `scale[token, h_block]`

### Pitfall 4: Full Weight Materialization

**Symptom:** OOM on large expert counts, slow throughput
**Cause:** Dequanting all 32 experts' weights to FP32 before compute
**Fix:** Dequant one expert at a time, or use on-the-fly dequant in GEMM tiles

### Pitfall 5: FP16 Accumulation Drift

**Symptom:** Growing error with large `T` (seq_len > 1000)
**Cause:** Accumulating GEMM products in FP16 instead of FP32
**Fix:** Always accumulate in FP32, cast to BF16 only at final output write

### Pitfall 6: Ignoring `local_expert_offset`

**Symptom:** Wrong experts selected, mismatched outputs
**Cause:** Not mapping global expert IDs to local expert indices
**Fix:** `local_idx = topk_idx - local_expert_offset`

---

## 11. CUDA Kernel Equivalents

The same logic is implemented in CUDA (`kernel.cu`) with these kernels:

| Triton Kernel | CUDA Equivalent | Grid/Block |
|---------------|----------------|------------|
| `_dequant_hidden_fp8_kernel` | `dequant_hidden_fp8` | `(⌈H/256⌉, ⌈T/32⌉)` blocks of `(256, 4)` |
| `_swiglu_kernel` | `swiglu_activation` | `(⌈I/256⌉, ⌈Tk/32⌉)` blocks of `(256, 4)` |
| — | `dequant_weight_fp8` | General FP8 weight dequant |
| — | `cast_fp32_to_bf16` | Final type cast |

**Key CUDA types:**
- `__nv_fp8_e4m3` — FP8 E4M3 type (from `cuda_fp8.h`)
- `__nv_bfloat16` — BF16 type (from `cuda_bf16.h`)
- `__float2bfloat16()` — FP32 → BF16 conversion intrinsic

---

## 12. References & Citations

[1] FlashInfer-Bench Solution Schema — Destination Passing Style.
    https://bench.flashinfer.ai/docs/flashinfer-trace/solution

[2] Triton Language Documentation.
    https://triton-lang.org/main/python-api/triton.html

[3] Triton Language — `program_id`.
    https://triton-lang.org/main/python-api/generated/triton.language.program_id.html

[4] Triton Language — `arange`.
    https://triton-lang.org/main/python-api/generated/triton.language.arange.html

[5] Triton Language — `load`.
    https://triton-lang.org/main/python-api/generated/triton.language.load.html

[6] Triton Language — `store`.
    https://triton-lang.org/main/python-api/generated/triton.language.store.html

[7] Triton Language — `sigmoid`.
    https://triton-lang.org/main/python-api/generated/triton.language.sigmoid.html

[8] Shazeer, N. "GLU Variants Improve Transformer." arXiv:2002.05202, 2020.
    https://arxiv.org/abs/2002.05202

[9] DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437, 2024.
    https://arxiv.org/abs/2412.19437
    → Section 3.3: No-Aux-Loss Routing with Bias

[10] NVIDIA CUDA Math API — FP8 Types (`__nv_fp8_e4m3`).
     https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__E4M3.html

[11] NVIDIA Blackwell Architecture Whitepaper, 2024.
     https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/

[12] Tillet, P., Kung, H.T., Cox, D. "Triton: An Intermediate Language and Compiler
     for Tiled Neural Network Computations." MLSys, 2019.
     https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf

[13] FlashInfer — Fused MoE Operations API.
     https://docs.flashinfer.ai/api/python/fused_moe.html

[14] PyTorch FP8 Documentation — `torch.float8_e4m3fn`.
     https://pytorch.org/docs/stable/tensors.html

[15] NVIDIA FP8 Formats for Deep Learning.
     https://arxiv.org/abs/2209.05433

[16] FlashInfer AI Kernel Generation Contest @ MLSys 2026.
     http://mlsys26.flashinfer.ai/

[17] DeepSeek-V3 Repository — `weight_dequant_kernel` (Triton reference).
     https://github.com/deepseek-ai/DeepSeek-V3

[18] vLLM Fused MoE Triton Kernels.
     https://docs.vllm.ai/en/latest/design/kernel/moe.html

[19] PyTorch Blog — 2D Dynamic Block Quantized Float8 GEMMs in Triton.
     https://pytorch.org/blog/dynamic-block-quantized-float8-gemms-in-triton/

---

> **Last updated:** 2026-03-11 | **Target:** NVIDIA B200 (Blackwell) | **Track:** fused_moe
