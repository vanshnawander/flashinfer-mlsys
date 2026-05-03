# B200 Register & Instruction Optimization Notes

## B200 Architecture (SM 10.0, Blackwell)
- **SM count**: 160 SMs
- **Registers per SM**: 65,536 (32-bit)
- **Max registers per thread**: 255
- **Shared memory per SM**: 228 KB
- **L2 cache**: 96 MB
- **Memory bandwidth**: 8 TB/s (HBM3e)
- **Compute**: FP32 = 90 TFLOPS, TF32 = 180 TFLOPS, FP8 = 2.25 PFLOPS

## Register Budget Analysis

### Per-Thread Register Pressure
Each thread gets ~255 max registers. Key variables that consume registers:

**Routing kernel** (element-wise, simple):
- 2 floats for sigmoid: `x`, `s` = 2 regs
- 1 float for bias lookup = 1 reg
- Total: ~5 registers (very light, can max occupancy)

**SwiGLU kernel** (element-wise):
- gate, up, sig = 3 floats = 3 regs
- Total: ~6 registers

**Dequant kernel** (if custom):
- fp8 input (1 reg), scale (1 reg), output (1 reg) = ~4 registers

**GEMM (Triton fused)**:
- Tile accumulators: BLOCK_M × BLOCK_N × sizeof(float) / 4 = 64×128×1 = 8192 values
- But stored across warps/threads: 8192 / 512 threads = 16 regs per thread
- + loop variables, pointers = ~5 regs
- + FP8 -> BF16 conversion temps = ~4 regs
- Total: ~30-40 registers → **allows 2 blocks per SM** (2 × 512 threads = 1024 threads)

### Occupancy Targets
| Registers/thread | Threads/SM | Blocks/SM | Occupancy |
|------------------|------------|-----------|-----------|
| 32 | 2048 | 4 | 100% |
| 48 | 1280 | 2 | 63% |
| 64 | 1024 | 2 | 50% |
| 128 | 512 | 1 | 25% |

**Goal**: Keep element-wise kernels < 32 regs (100% occupancy), GEMM < 64 regs (50%+ occupancy).

## Instruction-Level Optimizations

### Use CUDA Intrinsics Instead of ATen
| Operation | ATen (dispatch overhead) | CUDA Intrinsic (direct PTX) |
|-----------|-------------------------|-----------------------------|
| sigmoid(x) | `at::sigmoid()` — kernel launch + dispatch | `1.0f / (1.0f + __expf(-x))` — single instr |
| silu(x) | `at::silu()` — kernel launch + dispatch | `x / (1.0f + __expf(-x))` — 3 instrs |
| exp(x) | `at::exp()` | `__expf(x)` — fast approximation |
| rsqrt(x) | `at::rsqrt()` | `rsqrtf(x)` — HW instruction |

### `__expf` vs `expf`
- `__expf`: Uses PTX `ex2.approx` — **1 cycle**, ~2 ULP error
- `expf`: Full precision — **~8 cycles**
- For sigmoid/silu where we add 1.0, the approximation error is negligible

### Vectorized Memory Access
B200 HBM3e bandwidth: 8 TB/s. To saturate:
```
// 128-bit vectorized load (4 floats at once)
float4 data = *reinterpret_cast<const float4*>(&input[idx * 4]);
```
- Coalesced float4 loads → 4× fewer memory transactions
- Requires 16-byte alignment

### B200-Specific PTX Generation
Add to compile flags:
```
-gencode arch=compute_100,code=sm_100  // B200 specific
--use_fast_math                         // Enables __expf, __logf, etc.
-maxrregcount=64                        // Cap registers for GEMM occupancy
```

## What Can Be Custom-Fused

### 1. Fused Sigmoid + Bias (DONE ✅)
Before: `sigmoid()` kernel + `add()` kernel = 2 kernel launches, 2 memory passes
After: Single kernel, 1 memory pass, uses `__expf` intrinsic

### 2. Fused SwiGLU (DONE ✅)
Before: `slice()` + `silu()` + `mul()` = 3 kernel launches
After: Single kernel, reads g1 once, writes output once

### 3. Fused Weighted Scatter-Add (DONE ✅)
Before: `mul()` + `index_add_()` = 2 kernel launches
After: Single kernel with `atomicAdd`

### 4. Fused Dequant + GEMM (TODO - high impact)
Currently: `dequant_weight()` materializes full FP32 weight matrix, then cuBLAS GEMM
Opportunity: Stream FP8 weights through registers, dequant per-tile during GEMM
- Saves materializing (2*I × H) = 29M floats = 116MB per expert
- This is what the Triton fused kernel already does

### 5. Fused Bias-Dequant for Hidden States (TODO)
Currently: `.to(kFloat)` + `.view()` + `* scale` + `.reshape()`
Opportunity: Single kernel that reads FP8 + scale, writes FP32 in one pass

## Variable Waste Analysis

### Triton kernel.py — Fixed
- `logits = routing_logits.to(torch.float32)` — **WASTED**: routing_logits IS float32
- `use_bulk` conditional — **WASTED**: always bulk-gather now
- `topk_w ... .to(torch.float32)` — **WASTED**: computed in float32 already, just cast once

### CUDA kernel.cu — Fixed
- `routing_logits.to(at::kFloat)` — **REMOVED**: already float32
- Separate `at::sigmoid()` + `+ bias` — **FUSED**: single CUDA kernel
- `at::silu()` in SwiGLU — **REPLACED**: custom kernel with `__expf`
- Per-expert `index_select` — **REPLACED**: pre-gathered batch `sorted_a`, `sorted_w`

## Next Steps for Further Optimization
1. **Fused dequant GEMM kernel** — biggest remaining win for CUDA path
2. **Multi-stream expert processing** — overlap GEMM1 of expert i+1 with GEMM2 of expert i
3. **Persistent kernel** — single kernel launch processes all experts (eliminates launch overhead)
4. **SMEM tiling** — for dequant, keep FP8→FP32 scale in shared memory
5. **Warp specialization** — dedicated warps for memory vs compute (Blackwell feature)
