# MoE Kernel — CUDA & Triton Optimization Notes (v15)

## What Changed to Make CUDA Work on B200

### Root Cause of COMPILE_ERROR
The `flashinfer-bench` framework has a `TorchBuilder` that handles `language: "cuda"`. Three things were wrong:

1. **No `nvcc` on Modal image** — The starter-kit `debian_slim` image has CUDA runtime but **no CUDA toolkit** (no `nvcc`). The `TorchBuilder` uses `torch.utils.cpp_extension.load()` which needs `nvcc`.

2. **Missing `binding: "torch"` in BuildSpec** — `TorchBuilder.can_build()` checks `solution.spec.binding == SupportedBindings.TORCH`. Default is `None` which falls through to `TVMFFIBuilder`. Without `tvm_ffi` installed, no builder could handle our solution.

3. **Entry point was `.py` file** — `TorchBuilder` validates the entry file extension must be `.cu`/`.cpp`. Our old `binding.py::kernel` was rejected.

### Fixes Applied
| File | Change |
|------|--------|
| `scripts/run_modal.py` | Added CUDA toolkit (`cuda-nvcc-12-8`) + build tools to Modal image |
| `scripts/pack_solution.py` | Added `binding="torch"` when `language == "cuda"` |
| `config.toml` | Changed entry_point to `kernel.cu::kernel` |
| `solution/cuda/kernel.cu` | Full ATen C++ kernel — routing + compute, no Python needed |

### Architecture: ATen C++ Kernel
The CUDA kernel (`kernel.cu`) is a **pure C++ implementation** using ATen (PyTorch's C++ tensor library):
- **Routing logic**: ATen ops (`at::sigmoid`, `at::topk`, `at::gather`, etc.)
- **GEMMs**: `at::matmul` → cuBLAS under the hood
- **Dequant/SwiGLU**: ATen element-wise ops
- **Advantage**: Zero Python loop overhead for per-expert compute

### Key Detail: FP32 Precision
Added `at::globalContext().setFloat32MatmulPrecision("highest")` because B200 uses TF32 by default in cuBLAS, which truncates mantissa bits and causes large numerical errors (abs_err up to 4096).

---

## B200 Benchmark Results (v13 CUDA)

| Seq Len | Latency (ms) | Speedup | Abs Error | Notes |
|---------|-------------|---------|-----------|-------|
| 1 | 1.596 | **6.94x** | 0 | Perfect |
| 7 | 3.143 | **3.70x** | 0 | Perfect |
| 16 | 5.756 | **2.22x** | 0.125 | OK |
| 32 | 9.434 | 1.51x | 32 | Needs work |
| 80 | 13.333 | 1.21x | 1024 | Large error |
| 901 | 19.044 | 1.11x | 2048 | Large error |
| 14107 | 43.267 | 1.05x | 2048 | Barely faster |

**Key observation**: Small-T is great (7x speedup), large-T is barely faster than reference and has numerical issues.

---

## Optimization Plan

### Phase 1: Fix Numerical Accuracy
- [x] Disable TF32 in matmul → `setFloat32MatmulPrecision("highest")`
- [x] Use stable sort
- [ ] Verify FP8 dequant matches reference exactly
- [ ] Test on B200 with precision fix

### Phase 2: Triton Kernel Optimization
Current Triton kernel (`solution/triton/kernel.py`) has fused BF16 GEMM kernels that are faster for large-T. Focus areas:
- [ ] **Register pressure**: Remove duplicate variable loads in GEMM1 kernel
- [ ] **Tile size tuning**: BLOCK_M=64,BLOCK_N=128 may not be optimal for B200's 228KB SMEM
- [ ] **Pipeline stages**: Currently num_stages=3; try 4 or 5 for better latency hiding
- [ ] **Multi-stream**: Launch multiple experts concurrently for large-T

### Phase 3: NCU Profiling
Using the flashinfer-bench NCU API:
```bash
# Pack solution first
python scripts/pack_solution.py

# Run NCU profile (requires FIB_DATASET_PATH)
python scripts/run_ncu_profile.py                    # Profiles 3 representative workloads
python scripts/run_ncu_profile.py --workload-idx 0   # Profile specific workload
python scripts/run_ncu_profile.py --set full          # Full metric collection
```

Output saved to `ncu_profiles/<solution_name>_<timestamp>/`

### Phase 4: Hybrid CUDA + Triton
Best of both worlds:
- Small-T (decode, T<32): CUDA ATen kernel (low overhead)
- Large-T (prefill, T>100): Triton fused kernels (high throughput)

---

## File Structure
```
solution/
├── triton/
│   └── kernel.py          # Triton fused GEMM kernels (main submission)
└── cuda/
    ├── kernel.cu          # ATen C++ kernel (torch binding)
    └── binding.py         # Legacy Python binding (not used with TorchBuilder)

scripts/
├── run_modal.py           # Modal B200 benchmark (CUDA toolkit in image)
├── run_ncu_profile.py     # NCU profiler using flashinfer-bench API
├── pack_solution.py       # Packs solution.json (adds binding=torch for cuda)
└── run_local.py           # Local benchmark

ncu_profiles/              # NCU profile outputs (git-ignored)
```
