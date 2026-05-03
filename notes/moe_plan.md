# MoE Kernel Optimization Plan — v13 B200 Results

## Status: CUDA Kernel Running on B200 ✅

### What We Fixed
1. **Root cause of COMPILE_ERROR**: No `nvcc` on Modal image + missing `binding: "torch"` in BuildSpec
2. **Fix**: Added CUDA toolkit to Modal image, set `binding: "torch"`, entry point `kernel.cu::kernel`
3. **Summary counter bug**: Was checking `"success"` instead of `"PASSED"` — fixed

---

## B200 Results (v13 CUDA — ATen C++ kernel)

| Seq Len | Latency (ms) | Speedup | Abs Error | Rel Error | Status |
|---------|-------------|---------|-----------|-----------|--------|
| 1 | 1.596 | **6.94x** | 0 | 0 | ✅ |
| 7 | 3.143 | **3.70x** | 0 | 0 | ✅ |
| 14 | 5.159 | **2.43x** | 0.016 | 0.005 | ✅ |
| 15 | 2.561 | **4.49x** | 8 | 0.007 | ⚠️ |
| 16 | 5.756 | **2.22x** | 0.125 | 0.007 | ✅ |
| 32 | 9.434 | 1.51x | 32 | 0.007 | ⚠️ |
| 52 | 7.395 | 1.81x | 2048 | 0.056 | ❌ |
| 53 | 10.672 | 1.40x | 1024 | 0.008 | ❌ |
| 54 | 10.096 | 1.44x | 1024 | 0.008 | ❌ |
| 55 | 10.617 | 1.39x | 512 | 0.008 | ❌ |
| 56 | 11.492 | 1.33x | 1024 | 0.167 | ❌ |
| 57 | 11.285 | 1.35x | 512 | 0.014 | ❌ |
| 58 | 12.184 | 1.28x | 1024 | 0.013 | ❌ |
| 59 | 8.368 | 1.66x | 64 | 0.095 | ❌ |
| 62 | 8.251 | 1.69x | 512 | 0.012 | ❌ |
| 80 | 13.333 | 1.21x | 1024 | 0.138 | ❌ |
| 901 | 19.044 | 1.11x | 2048 | 1.0 | ❌ |
| 11948 | 33.581 | 1.07x | 4096 | 2.33 | ❌ |
| 14107 | 43.267 | 1.05x | 2048 | **879000** | ❌ |

---

## Critical Issues

### 1. Numerical Errors (PRIORITY 1)
Errors grow with T. Root cause candidates:
- **FP8→FP32 dequant mismatch**: ATen's `.to(kFloat)` for FP8 may differ from reference's blockscale dequant
- **Pre-dequanting ALL tokens**: We dequant everything upfront, reference may dequant per-expert
- **cuBLAS precision**: `at::matmul` may use different precision (TF32 vs FP32) than reference
- **`at::sort` stability**: Non-stable sort could reorder tokens within same expert, causing accumulation drift

### 2. Large-T Performance (PRIORITY 2)  
- T=14107: only 1.05x speedup (43ms vs 45ms reference)
- The C++ loop eliminates Python overhead but ATen ops are not fused
- Need Triton fused kernels for large-T experts

---

## Optimization Plan

### Phase 1: Fix Numerical Errors
1. Match reference dequant exactly (blockscale FP8 with correct scale indexing)
2. Use stable sort (`at::sort` with `stable=true`)
3. Force FP32 matmul precision (disable TF32)

### Phase 2: Triton Kernel Optimization
1. Run NCU profiling via `flashinfer_bench_run_ncu()` API
2. Reduce register pressure in fused GEMM1+SwiGLU kernel
3. Tune tile sizes for B200's 228KB SMEM
4. Optimize large-T path (multi-stream, persistent kernel)

### Phase 3: NCU Profiling (per starter kit README)
```python
from flashinfer_bench.agents import flashinfer_bench_run_ncu
output = flashinfer_bench_run_ncu(
    solution=solution, workload=workload,
    set="detailed", page="details", timeout=120
)
```
