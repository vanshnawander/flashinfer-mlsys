# Fused MoE Kernel — Bug Analysis, Fixes & B200 Optimization Report

> Analysis of the Triton kernel failure, applied fixes, CUDA implementation, and
> performance bottlenecks for the FlashInfer MLSys'26 fused_moe track on NVIDIA B200.

---

## 🔴 Root Cause: Why the Kernel Was Failing

### Primary Bug: DPS Signature Mismatch

> [!CAUTION]
> This was the **primary failure mode** — the kernel crashed before any computation.

The `solution.json` has `"destination_passing_style": true` (the default in `BuildSpec`):

```json
{
  "spec": {
    "destination_passing_style": true,
    ...
  }
}
```

This tells the FlashInfer-Bench framework to call the kernel as:

```python
kernel(routing_logits, routing_bias, ..., routed_scaling_factor, output)
#                                                                ^^^^^^
#                                          pre-allocated [T, H] bf16 tensor
```

But the **original kernel signature** was:

```python
def kernel(routing_logits, routing_bias, ..., routed_scaling_factor):
    ...
    return output.to(torch.bfloat16)  # ← value-returning style
```

**Result:** `TypeError` — the framework passes 11 arguments but the kernel only accepts 10.

### Fix Applied

```diff
 def kernel(
     routing_logits, routing_bias, hidden_states, hidden_states_scale,
     gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
     local_expert_offset, routed_scaling_factor,
+    output: torch.Tensor,  # DPS: pre-allocated [T, H] bf16
 ):
     ...
-    return output.to(torch.bfloat16)
+    output.copy_(accum.to(torch.bfloat16))  # write in-place
```

---

## ✅ Correctness Verification

### Comparison with Reference ([ref_moe.py](file:///home/vanshnawander/accelerated-hpc/flashinfer-mlsys/reference%20moe/ref_moe.py))

| Stage | Kernel Implementation | Reference | Match? |
|-------|----------------------|-----------|--------|
| FP8 dequant (hidden) | Triton kernel: `x * scale[h//128, t]` | `A_fp32 * A_scale_expanded` | ✅ |
| FP8 dequant (W13) | Block reshape: `view(n_out, 128, n_h, 128) * s` | `repeat_interleave(S13, 128)` | ✅ Equivalent |
| FP8 dequant (W2) | Same block reshape pattern | Same repeat_interleave | ✅ Equivalent |
| Routing: sigmoid | `torch.sigmoid(logits)` | `1/(1+exp(-logits))` | ✅ Identical |
| Routing: bias | `s + bias` | `s + bias` | ✅ |
| Routing: group top-2 | `topk(s_wb_grouped, k=2)` | Same | ✅ |
| Routing: group top-4 | `topk(group_scores, k=4)` | Same | ✅ |
| Routing: global top-8 | `topk(scores_pruned, k=8)` | Same | ✅ |
| Routing: normalize | `gather(s, topk_idx) / sum` | `s * M / sum` | ✅ Equivalent |
| SwiGLU | `x1 * (x2 * sigmoid(x2))` | `silu(x2) * x1` | ✅ Identical |
| GEMM1 | `a_e @ w13.t()` | `A_e @ W13_e.t()` | ✅ |
| GEMM2 | `c @ w2.t()` | `C @ W2_e.t()` | ✅ |
| Weighted accum | `index_add_(0, tok_idx, o * w)` | Same | ✅ |
| Output dtype | `bf16 (copy_)` | `bf16 (return)` | ✅ |

> [!NOTE]
> One subtle difference: the reference creates a full `[T, E_global]` weighted mask and
> uses `.index_select` per global expert, while the kernel uses `topk_w[token_idx, topk_pos]`
> with the compact `[T, 8]` topk arrays. Both are mathematically equivalent.

---

## 📊 Performance Bottlenecks

### Current Bottleneck Analysis

```mermaid
graph LR
    A[FP8 Dequant Hidden] -->|Fast: Triton kernel| B[Routing]
    B -->|Medium: PyTorch ops| C[Expert Loop x32]
    C -->|SLOW: Sequential| D[Weight Dequant]
    D -->|SLOW: Full materialization| E[GEMM1]
    E --> F[SwiGLU]
    F -->|Fast: Triton kernel| G[GEMM2]
    G --> H[Weighted Accum]
    H -->|Fast: index_add_| I[Output]
    
    style C fill:#ff6b6b
    style D fill:#ff6b6b
    style E fill:#ffa07a
    style G fill:#ffa07a
```

### Bottleneck 1: Sequential Expert Loop 🔴

**Problem:** 32 experts are processed sequentially in a Python `for` loop.

**Impact:** 32× kernel launch overhead, no cross-expert parallelism.

**Fix options:**
- **Token reordering + batched GEMM:** Sort tokens by expert, use one batched GEMM
- **Grouped GEMM:** Use Triton `tl.dot()` with dynamic routing
- **Persistent kernel:** Process all experts in one kernel launch

### Bottleneck 2: Full Weight Materialization 🔴

**Problem:** Each expert's weights are dequantized to a full FP32 tensor before GEMM.
- W13: `4096 × 7168 × 4B` = **112 MB per expert**, 3.5 GB total
- W2: `7168 × 2048 × 4B` = **56 MB per expert**, 1.8 GB total

**Impact:** Wastes HBM bandwidth and memory.

**Fix:** On-the-fly dequant in GEMM tiles — load FP8 tile, apply scale from SMEM, feed to MMA.

### Bottleneck 3: PyTorch GEMM Instead of Triton 🟡

**Problem:** Using `torch.matmul()` for GEMMs — goes through cuBLAS, decent but not optimal.

**Fix:** Custom Triton GEMM with fused dequant → eliminates separate dequant pass entirely.

### Bottleneck 4: Routing in PyTorch 🟡

**Problem:** 7 PyTorch kernel launches just for routing (`sigmoid`, `view`, `topk` × 3, `scatter_`, `gather`).

**Fix:** Single Triton routing kernel processing 256 experts per token.

---

## 🎯 Optimization Roadmap for B200

### Phase 1: Correctness ✅ (Done)
- [x] Fix DPS signature
- [x] Match reference routing algorithm
- [x] FP32 accumulation throughout
- [x] BF16 cast only at output

### Phase 2: Low-Hanging Fruit
- [ ] Token reordering by expert (coalesced access)
- [ ] Triton routing kernel (reduce launch count)
- [ ] `torch.compile` on the Python routing code

### Phase 3: Fused Kernels
- [ ] Fused GEMM1 + dequant (Triton `tl.dot()` with FP8 tiles)
- [ ] Fused GEMM1 + SwiGLU + GEMM2 (persistent block kernel)
- [ ] Grouped GEMM across experts

### Phase 4: B200-Specific
- [ ] Tune tile sizes for 228 KB SMEM (larger K tiles)
- [ ] FP8 tensor core MMA instructions
- [ ] Thread block clusters for multi-SM cooperation
- [ ] L2 cache partitioning for weight reuse

### Expected Speedups

| Optimization | Estimated Speedup | Effort |
|-------------|-------------------|--------|
| Token reordering | 1.3–1.5× | Low |
| Fused GEMM+dequant | 2–3× | Medium |
| Persistent expert kernel | 1.5–2× | Medium |
| Full fusion pipeline | 5–10× | High |
| B200 SMEM tuning | 1.2–1.5× | Medium |

---

## 📁 Files Modified

| File | Change | Reason |
|------|--------|--------|
| [kernel.py (Triton)](file:///home/vanshnawander/accelerated-hpc/flashinfer-mlsys/solution/triton/kernel.py) | Added `output` DPS param, `output.copy_()` | Fix primary failure |
| [kernel.cu](file:///home/vanshnawander/accelerated-hpc/flashinfer-mlsys/solution/cuda/kernel.cu) | Full CUDA kernels for dequant, SwiGLU, cast | New CUDA implementation |
| [binding.py](file:///home/vanshnawander/accelerated-hpc/flashinfer-mlsys/solution/cuda/binding.py) | Full Python binding with DPS | New CUDA binding |
| [triton_kernel_concepts.md](file:///home/vanshnawander/accelerated-hpc/flashinfer-mlsys/reference%20moe/triton_kernel_concepts.md) | Comprehensive Triton reference | New documentation |
