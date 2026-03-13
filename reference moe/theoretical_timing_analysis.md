# Theoretical vs Actual Timing Analysis

> For each operation in the kernel, compute the theoretical minimum time on B200
> hardware, then compare against actual benchmarks to find the gaps.

---

## B200 Hardware Specs (used for all calculations)

```
FP32 CUDA cores:      ~60 TFLOPS  (18,944 cores × ~3.2 GHz)
TF32 Tensor cores:    ~120 TFLOPS (using torch.matmul)
FP8 Tensor cores:     ~4,000 TFLOPS (native FP8 MMA if available)
BF16 Tensor cores:    ~2,000 TFLOPS

HBM3e bandwidth:      8,000 GB/s (8 TB/s)
L2 cache:             126 MB
SMEM per SM:          228 KB
SMs:                  148

Kernel launch overhead: ~5-15 µs per launch (CUDA kernel)
Python overhead:        ~1-5 µs per Python statement
torch op overhead:      ~10-30 µs per PyTorch operation (includes dispatch)
```

---

## Fixed Parameters

```
H  = 7168   (hidden size)
I  = 2048   (intermediate size)
2I = 4096   (GEMM1 output = W1 + W3 concatenated)
E  = 32     (local experts)
K  = 8      (top-k experts per token)
BLOCK_Q = 128  (FP8 quantization block)
```

---

## Per-Token Data Sizes

```
hidden_states:        7168 × 1B (FP8) = 7 KB per token
hidden_states_scale:  56 × 4B (FP32)  = 224 B per token
routing_logits:       256 × 4B (FP32)  = 1 KB per token
routing_bias:         256 × 2B (BF16)  = 512 B (shared, not per-token)
```

---

## Per-Expert Weight Sizes

```
W13 (FP8):     4096 × 7168 × 1B  = 28.0 MB
W13 scale:     32 × 56 × 4B      = 7.0 KB
W2 (FP8):      7168 × 2048 × 1B  = 14.0 MB
W2 scale:      56 × 16 × 4B      = 3.5 KB
Total per expert: 42.0 MB (FP8 weights only)
```

---

## Stage-by-Stage Theoretical Analysis

### Stage 1: FP8 Dequant Hidden States

**What:** `a[t,h] = hidden_states[t,h] × scale[h÷128, t]`

**Data movement:**
```
Read:  T × 7168 × 1B  (FP8 hidden states)  = T × 7.0 KB
Read:  56 × T × 4B    (scales, transposed)  = T × 0.22 KB
Write: T × 7168 × 4B  (FP32 output)         = T × 28.0 KB
Total: T × 35.2 KB
```

**Compute:** T × 7168 multiplications = T × 7168 FLOP (negligible)

**This is MEMORY-BOUND.**

| T | Data (MB) | Theoretical Time | Bottleneck |
|---|-----------|-----------------|------------|
| 1 | 0.034 | 0.004 µs | Launch overhead >> data |
| 8 | 0.275 | 0.034 µs | Launch overhead >> data |
| 64 | 2.2 | 0.28 µs | Launch overhead >> data |
| 256 | 8.8 | 1.1 µs | Becoming memory-bound |
| 1024 | 35.2 | 4.4 µs | Memory-bound |
| 2048 | 70.4 | 8.8 µs | Memory-bound |

**Actual estimate:** ~0.05-0.1 ms (dominated by Triton kernel launch overhead)

---

### Stage 2: Routing

**Operations (each is one CUDA kernel launch):**

| # | Operation | Data (for T=256) | FLOP | Time (compute) | Time (launch) |
|---|-----------|-----------------|------|---------------|---------------|
| 1 | `logits.to(fp32)` | 256×256×4B = 256KB | 0 | 0.03 µs | ~15 µs |
| 2 | `bias.to(fp32).view(-1)` | 256×2B = 512B | 0 | ~0 | ~15 µs |
| 3 | `torch.sigmoid(logits)` | 256KB in + 256KB out | 256K FLOP | 0.004 µs | ~15 µs |
| 4 | `s + bias` (broadcast) | 256KB + 1KB → 256KB | 65K FLOP | ~0 | ~15 µs |
| 5 | `.view(T, 8, 32)` | 0 (metadata) | 0 | 0 | 0 |
| 6 | `topk(k=2, dim=2)` | 256×8×32 | small | ~0 | ~15 µs |
| 7 | `.sum(dim=2)` | 256×8×2 → 256×8 | 4K FLOP | ~0 | ~15 µs |
| 8 | `topk(k=4, dim=1)` | 256×8 | small | ~0 | ~15 µs |
| 9 | `zeros_like + scatter_` | 256×8 | 0 | ~0 | ~30 µs (2 ops)|
| 10 | `unsqueeze + expand + reshape` | 0 (metadata) | 0 | 0 | 0 |
| 11 | `masked_fill` | 256×256 | 0 | ~0 | ~15 µs |
| 12 | `topk(k=8, dim=1)` | 256×256 | small | ~0 | ~15 µs |
| 13 | `gather` | 256×8 from 256×256 | 0 | ~0 | ~15 µs |
| 14 | `sum + div + mul` | 256×8 | 5K FLOP | ~0 | ~45 µs (3 ops)|

**Total routing:**
```
Compute time:  ~0 (all tensors tiny, negligible)
Launch overhead: ~13 ops × 15 µs = ~195 µs ≈ 0.2 ms
```

**This is LAUNCH-OVERHEAD-BOUND.**

| T | Compute | Launch Overhead | Total Theoretical |
|---|---------|----------------|-------------------|
| 1 | ~0 | ~0.2 ms | ~0.2 ms |
| 64 | ~0 | ~0.2 ms | ~0.2 ms |
| 256 | ~0 | ~0.2 ms | ~0.2 ms |
| 1024 | ~0.001 ms | ~0.2 ms | ~0.2 ms |
| 2048 | ~0.002 ms | ~0.2 ms | ~0.2 ms |

**Key insight:** Routing takes ~0.2 ms regardless of T because it's just launch overhead.
For T=1 workloads taking 1.7 ms total, routing is ~12% of time.

---

### Stage 3: Dispatch Table

**Operations:**
```
1. local_idx = topk_idx - local_start       ~15 µs (sub + compare)
2. valid_local = (>=0) & (<32)              ~15 µs
3. torch.any(valid_local)                   ~15 µs
4. torch.nonzero(valid_local)               ~15 µs
5. indexing for flat_expert_id              ~15 µs
6. torch.argsort(flat_expert_id)            ~15 µs
7. indexing for sorted tensors              ~15 µs
8. torch.unique_consecutive                 ~15 µs
9. torch.cumsum(counts)                     ~15 µs
10. Python loop (boundaries)                ~5 µs × num_active_experts
```

**Total dispatch:** ~0.14 ms + Python loop overhead (~0.15 ms)

---

### Stage 4: Bulk Gather (NEW in sub-6)

**What:** `sorted_a = a.index_select(0, sorted_token_idx)`

**Data movement:**
```
N_valid = T × K × (E_local / E_global) ≈ T × 8 × (32/256) = T × 1
  (on average, each token hits ~1 local expert)
  Actually: T × 8 assignments, ~T local ones

Read:  N_valid × 7168 × 4B = N_valid × 28 KB
Write: N_valid × 7168 × 4B = N_valid × 28 KB
Total: N_valid × 56 KB
```

| T | N_valid (approx) | Data | Theoretical |
|---|-----------------|------|-------------|
| 1 | ~1 | 56 KB | ~0 (launch overhead) |
| 64 | ~64 | 3.5 MB | 0.4 µs |
| 256 | ~256 | 14 MB | 1.7 µs |
| 1024 | ~1024 | 56 MB | 7.0 µs |
| 2048 | ~2048 | 112 MB | 14.0 µs |

**Actual:** ~0.02 ms (one kernel launch + memory copy)

---

### Stage 5: Expert Compute (THE BOTTLENECK)

For ONE expert with Tk tokens:

#### GEMM1: `[Tk, 7168] × [7168, 4096]ᵀ → [Tk, 4096]`

**Compute:**
```
FLOP = Tk × 7168 × 4096 × 2 = Tk × 58,720,256
     ≈ Tk × 58.7 MFLOP
```

**Data movement (fused path — reads FP8 weights):**
```
Read A:     Tk × 7168 × 4B   = Tk × 28 KB          (from sorted_a, contiguous)
Read W13:   4096 × 7168 × 1B = 28 MB               (FP8 weights from HBM)
Read scale: 32 × 56 × 4B     = 7 KB                 (negligible)
Write out:  Tk × 4096 × 4B   = Tk × 16 KB          (GEMM1 output)
Total: 28 MB + Tk × 44 KB
```

**Data movement (cuBLAS fallback — reads FP32 weights):**
```
Read W13 FP8: 4096 × 7168 × 1B = 28 MB
Write W13 FP32: 4096 × 7168 × 4B = 112 MB            ← THIS IS THE COST
Read W13 FP32: 4096 × 7168 × 4B = 112 MB             (cuBLAS reads it)
Total: 252 MB + Tk × 44 KB
                ↑
        9× more bandwidth than fused path!
```

| Tk | FLOP | Compute (FP32 60T) | Compute (TF32 120T) | Bandwidth (fused) | Bandwidth (cuBLAS) |
|----|------|--------------------|--------------------|-------------------|--------------------|
| 1 | 58.7M | 0.001 ms | 0.0005 ms | 28 MB → 3.5 µs | 252 MB → 31.5 µs |
| 8 | 470M | 0.008 ms | 0.004 ms | 28 MB → 3.5 µs | 252 MB → 31.5 µs |
| 32 | 1.88G | 0.031 ms | 0.016 ms | 29 MB → 3.6 µs | 253 MB → 31.6 µs |
| 128 | 7.52G | 0.125 ms | 0.063 ms | 34 MB → 4.2 µs | 258 MB → 32.2 µs |
| 256 | 15.0G | 0.250 ms | 0.125 ms | 39 MB → 4.9 µs | 263 MB → 32.9 µs |

**Bottleneck analysis per expert GEMM1:**
- **Tk < 32:** Memory-bound (weight loading dominates). Fused = 3.5µs, cuBLAS = 31.5µs. **Fused is 9× better.**
- **Tk ≥ 128:** Compute-bound. FP32 CUDA cores = 0.125ms, TF32 = 0.063ms. **TF32 is 2× better but we can't use it with fused!**
- **Sweet spot for fused:** Tk < ~100 where bandwidth savings > compute loss.

#### SwiGLU: `up × silu(gate) → [Tk, 2048]`

**Compute:** `Tk × 2048 × 3 = Tk × 6144 FLOP` (sigmoid + multiply × 2)
**Data:** Read `Tk × 4096 × 4B`, Write `Tk × 2048 × 4B` = `Tk × 24 KB`

| Tk | FLOP | Compute | Bandwidth | Bottleneck |
|----|------|---------|-----------|------------|
| 32 | 197K | ~0 | 768 KB → 0.1 µs | Launch overhead |
| 256 | 1.6M | ~0 | 6.1 MB → 0.8 µs | Launch overhead |
| 1024 | 6.3M | ~0 | 24 MB → 3.0 µs | Memory-bound |

**SwiGLU is always negligible** (< 0.01 ms even for large Tk).

#### GEMM2: `[Tk, 2048] × [2048, 7168]ᵀ → [Tk, 7168]`

**Compute:** `Tk × 2048 × 7168 × 2 = Tk × 29.4 MFLOP`
**Data (fused):** Read W2 FP8: 14 MB + Tk × 36 KB

| Tk | FLOP | Compute (FP32) | Bandwidth (fused) |
|----|------|---------------|-------------------|
| 1 | 29.4M | 0.0005 ms | 14 MB → 1.75 µs |
| 32 | 940M | 0.016 ms | 15 MB → 1.9 µs |
| 256 | 7.52G | 0.125 ms | 23 MB → 2.9 µs |

GEMM2 is ~half the cost of GEMM1 (weight matrix is 2× smaller).

#### Scatter-add: `accum.index_add_(0, t_idx, o × w)`

**Data:** Read + Write: `Tk × 7168 × 4B × 2 = Tk × 56 KB`
**Time:** `Tk × 56 KB / 8 TB/s ≈ Tk × 7 ns`
Plus launch overhead: ~15 µs

---

### Stage 5 Total: ALL 32 Experts

**Theoretical minimum per expert (Tk=32, fused path):**
```
GEMM1:         max(0.031 ms compute, 0.004 ms bandwidth) = 0.031 ms
SwiGLU:        ~0 ms
GEMM2:         max(0.016 ms compute, 0.002 ms bandwidth) = 0.016 ms
Scatter-add:   0.015 ms (launch overhead)
Expert total:  0.062 ms
```

**Theoretical minimum for 32 experts (sequential):**
```
32 × 0.062 ms = 1.98 ms
+ 32 × 3 × 0.010 ms (launch overhead per kernel) = 0.96 ms
Total: 2.94 ms
```

**Theoretical minimum for 32 experts (if we could parallelize all):**
```
One big GEMM across all experts: 32 × 0.062 ms / 148 SMs ≈ 0.013 ms
(This requires grouped GEMM or persistent kernel — not yet implemented)
```

---

## Full Pipeline Theoretical Timing

### For T=1 (decode-like, smallest workload)

```
Stage 1: FP8 dequant          0.05 ms  (Triton launch overhead)
Stage 2: Routing              0.20 ms  (13 PyTorch ops × 15µs)
Stage 3: Dispatch             0.15 ms  (8 PyTorch ops + Python loop)
Stage 4: Bulk gather          0.02 ms  (one index_select)
Stage 5: Expert loop          0.50 ms  (few active experts, small Tk)
Stage 6: Output cast+copy     0.02 ms

THEORETICAL TOTAL:            0.94 ms
ACTUAL (sub-3 best):          1.72 ms  (e05c6c03)
GAP:                          1.83×

WHERE IS THE GAP?
  → Python interpreter overhead in expert loop (~0.3 ms)
  → PyTorch op dispatch overhead (each torch call ~20µs, not 15µs)
  → CUDA driver overhead for small kernels
```

### For T=64 (medium workload)

```
Stage 1: FP8 dequant          0.05 ms
Stage 2: Routing              0.20 ms
Stage 3: Dispatch             0.15 ms
Stage 4: Bulk gather          0.02 ms
Stage 5: Expert loop (Tk≈16 avg per expert)
  Per expert:
    GEMM1: max(0.015ms compute, 0.004ms bw) = 0.015 ms
    SwiGLU: ~0 ms
    GEMM2: max(0.008ms compute, 0.002ms bw) = 0.008 ms
    Launches: 3 × 0.015 ms = 0.045 ms
    Total per expert: 0.068 ms
  32 experts × 0.068 = 2.18 ms
Stage 6: Output               0.02 ms

THEORETICAL TOTAL:            2.62 ms
ACTUAL (sub-3, ~5-6 ms workloads): ~5.5 ms
GAP:                          2.1×

WHERE IS THE GAP?
  → 32 × torch.matmul launch + compute overhead
  → Weight dequant for cuBLAS fallback (Tk<32 → full 112 MB materialize per expert)
  → FIX: Lower fused threshold to catch these Tk=16 experts
```

### For T=256 (medium-large workload)

```
Stage 1: FP8 dequant          0.05 ms
Stage 2: Routing              0.22 ms
Stage 3: Dispatch             0.15 ms
Stage 4: Bulk gather          0.03 ms
Stage 5: Expert loop (Tk≈64 avg per expert)
  Per expert:
    GEMM1: max(0.063ms compute, 0.004ms bw) = 0.063 ms
    SwiGLU: ~0 ms
    GEMM2: max(0.031ms compute, 0.002ms bw) = 0.031 ms
    Launches: 3 × 0.015 ms = 0.045 ms
    Total per expert: 0.139 ms
  32 experts × 0.139 = 4.45 ms
Stage 6: Output               0.03 ms

THEORETICAL TOTAL:            4.93 ms
ACTUAL (sub-3, ~10-12 ms workloads): ~10.5 ms
GAP:                          2.1×

WHERE IS THE GAP?
  → Same pattern: actual GEMM is ~2× slower than theoretical
  → Our fused tl.dot on FP32 CUDA cores at 60 TFLOPS is slower than
    theoretical because of:
      a) Not all SMs are utilized (occupancy < 100%)
      b) Memory stalls (scale loading, non-coalesced access)
      c) Kernel launch overhead between experts
```

### For T=1024 (large workload)

```
Stage 1: FP8 dequant          0.06 ms
Stage 2: Routing              0.25 ms
Stage 3: Dispatch             0.16 ms
Stage 4: Bulk gather          0.05 ms
Stage 5: Expert loop (Tk≈256 avg per expert)
  Per expert:
    GEMM1: max(0.250ms compute, 0.005ms bw) = 0.250 ms
    SwiGLU: 0.003 ms
    GEMM2: max(0.125ms compute, 0.003ms bw) = 0.125 ms
    Launches: 3 × 0.015 ms = 0.045 ms
    Total per expert: 0.423 ms
  32 experts × 0.423 = 13.54 ms
Stage 6: Output               0.05 ms

THEORETICAL TOTAL:            14.11 ms
ACTUAL (sub-3, ~17-24 ms workloads): ~20 ms
GAP:                          1.4×

WHERE IS THE GAP?
  → Now mostly compute-bound in GEMM
  → FP32 CUDA cores at ~60 TFLOPS (theoretical is already high)
  → Achievable: ~70-80% of peak FP32 throughput = reasonable
  → BIG WIN: If we could use FP8 tensor cores (4000 TFLOPS):
      GEMM1: 0.250ms → 0.004ms (67× faster!)
      Total: 0.16 ms for all experts → ~1 ms total
```

---

## The Three Key Gaps

### Gap 1: Launch Overhead (dominates T < 32)

```
Fixed cost per forward pass:
  Routing:  ~13 PyTorch ops × 20 µs = 0.26 ms
  Dispatch: ~9 PyTorch ops × 20 µs  = 0.18 ms
  Expert loop: 32 experts × 3 Triton kernels × 15 µs = 1.44 ms
  Total fixed overhead: ~1.88 ms

This is WHY small-T workloads can never go below ~1.7 ms even though
the actual compute for T=1 is < 0.001 ms.

FIX: Fuse routing into 1 Triton kernel (saves 0.26 ms)
FIX: Grouped GEMM for all experts in 1 launch (saves 1.44 ms)
BEST POSSIBLE: ~0.2 ms total fixed overhead
```

### Gap 2: FP32 vs FP8 Compute (dominates T > 256)

```
Our fused GEMM uses tl.dot(fp32, fp32) → CUDA cores at ~60 TFLOPS
If we could use tl.dot(fp8, fp8) → Tensor cores at ~4000 TFLOPS

For T=1024, 32 experts:
  FP32: 32 × (0.250 + 0.125) = 12.0 ms compute
  FP8:  32 × (0.004 + 0.002) = 0.19 ms compute
  SAVING: 11.8 ms (!!!)

FIX: Use native FP8 tl.dot (requires Triton support for FP8 tensor cores)
STATUS: Triton 3.x+ supports tl.dot on fp8 directly on Blackwell
```

### Gap 3: Sequential Expert Loop (always present)

```
32 experts processed one at a time.
B200 has 148 SMs. If each expert uses ~4-8 SMs, we waste ~140 SMs.

For T=256 (Tk=64 per expert):
  Sequential: 32 × 0.139 ms = 4.45 ms
  Parallel (2 experts at a time): 16 × 0.139 ms = 2.22 ms
  Parallel (all 32): 0.139 ms (but need grouped GEMM)
  Persistent kernel: ~0.2 ms (all experts fused into one launch)

FIX: Grouped GEMM via CUTLASS or Triton persistent kernel
SAVING: 2-4× on expert compute
```

---

## Theoretical Minimum vs Actual (Summary Table)

| Workload Size | Theoretical Min | Best Actual (sub-3) | Gap | Dominant Bottleneck |
|--------------|-----------------|--------------------|----|---------------------|
| T=1 | 0.35 ms | 1.72 ms | 4.9× | Launch overhead (1.88 ms fixed) |
| T=8 | 0.45 ms | 2.63 ms | 5.8× | Launch overhead |
| T=64 | 2.62 ms | 5.50 ms | 2.1× | Weight materialization (cuBLAS fallback) |
| T=256 | 4.93 ms | 10.50 ms | 2.1× | FP32 compute + sequential loop |
| T=1024 | 14.11 ms | 20.00 ms | 1.4× | FP32 compute (approaching peak) |
| T=2048 | 27.5 ms | 24.36 ms | 0.9× | Already near theoretical FP32 peak! |

### Reaching 50% of theoretical peak

```
For 50% theoretical utilization on ALL workloads:

T=1:    need 0.70 ms → requires fusing routing + grouped expert kernel
T=8:    need 0.90 ms → same as above
T=64:   need 5.24 ms → lower fused threshold (sub-6 does this)
T=256:  need 9.86 ms → better tiling + pipelining (sub-6 does this)
T=1024: need 28.2 ms → already achieved! (sub-3 = 20 ms < 28.2 ms)

The hardest cases are T=1 and T=8, where launch overhead
is 4× the actual compute. Only solution: fuse everything.
```

---

## What Would Close Each Gap

| Gap | Fix | Expected Speedup | Difficulty |
|-----|-----|-----------------|------------|
| Launch overhead | Triton routing kernel | -0.26 ms | Medium |
| Launch overhead | Grouped GEMM (all experts in 1 launch) | -1.44 ms | Hard |
| Weight bandwidth | Keep fused GEMM, lower threshold | -0.5 ms mid-T | Easy ✅ (sub-6) |
| FP32 compute | FP8 tensor core GEMM | **10-60× on GEMM** | Hard (Triton FP8 tl.dot) |
| Sequential loop | Persistent MoE kernel | 2-4× on expert compute | Very Hard |
| Memory alloc | Pre-allocated buffers | -0.5 ms | Easy ✅ (sub-6) |
| Python overhead | torch.compile or CUDA graph | -0.3 ms | Medium |

---

> **Last updated:** 2026-03-13 | Based on B200 specs + sub-1/2/3/5 benchmarks
