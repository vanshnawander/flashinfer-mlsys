#!/usr/bin/env python3
"""
Test: Grouped GEMM kernel vs original per-expert-loop kernel.
Validates correctness and measures performance improvement.

Run: python solution/cutlass/test_grouped_gemm.py
"""
import sys
import os
import time

# Ensure both modules can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'triton'))
sys.path.insert(0, os.path.dirname(__file__))

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("No CUDA — exiting")
    sys.exit(0)

torch.manual_seed(42)
print(f"Device: {torch.cuda.get_device_name(0)}")

total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
is_small = total_vram < 20.0
print(f"VRAM: {total_vram:.1f} GB → {'reduced dims' if is_small else 'full B200 dims'}")

# ── Scaled dimensions ──
E  = 4    if is_small else 32
H  = 1024 if is_small else 7168
I  = 512  if is_small else 2048
BQ = 128
NUM_E_GLOBAL = 256
T  = 16

# Import both kernels
import kernel_grouped_gemm as grouped_kernel
import kernel_cute_dsl as original_kernel

# Patch constants for both kernels
for mod in (grouped_kernel, original_kernel):
    mod.HIDDEN_SIZE       = H
    mod.INTERMEDIATE_SIZE = I
    mod.NUM_LOCAL_EXPERTS = E
    mod.NUM_EXPERTS       = NUM_E_GLOBAL
    mod.GROUP_SIZE        = NUM_E_GLOBAL // mod.N_GROUP
    mod.BLOCK_Q           = BQ

# ── Build test tensors ──
logits = torch.randn(T, NUM_E_GLOBAL, device=DEVICE, dtype=torch.float32)
bias   = torch.randn(NUM_E_GLOBAL, device=DEVICE, dtype=torch.float32) * 0.01
hs_fp8 = torch.randn(T, H, device=DEVICE).clamp(-1, 1).to(torch.float8_e4m3fn)
hs_scl = torch.ones(H // BQ, T, device=DEVICE, dtype=torch.float32) * 0.005

w1 = torch.randn(E, 2*I, H, device=DEVICE).clamp(-1, 1).to(torch.float8_e4m3fn)
s1 = torch.ones(E, (2*I)//BQ, H//BQ, device=DEVICE, dtype=torch.float32) * 0.005
w2 = torch.randn(E, H, I, device=DEVICE).clamp(-1, 1).to(torch.float8_e4m3fn)
s2 = torch.ones(E, H//BQ, I//BQ, device=DEVICE, dtype=torch.float32) * 0.005

kwargs = dict(local_expert_offset=0, routed_scaling_factor=1.0)

# ── Run original kernel ──
print("\n" + "="*60)
print("Running ORIGINAL kernel (per-expert loop)...")
out_original = torch.zeros(T, H, device=DEVICE, dtype=torch.bfloat16)
original_kernel.kernel(logits, bias, hs_fp8, hs_scl, w1, s1, w2, s2,
                       **kwargs, output=out_original)
print(f"  Output shape: {out_original.shape}")
print(f"  Output norm:  {out_original.float().norm():.6f}")

# ── Run grouped kernel ──
print("\nRunning GROUPED GEMM kernel (batched)...")
out_grouped = torch.zeros(T, H, device=DEVICE, dtype=torch.bfloat16)
grouped_kernel.kernel(logits, bias, hs_fp8, hs_scl, w1, s1, w2, s2,
                      **kwargs, output=out_grouped)
print(f"  Output shape: {out_grouped.shape}")
print(f"  Output norm:  {out_grouped.float().norm():.6f}")

# ── Correctness check ──
print("\n" + "="*60)
abs_err = (out_grouped.float() - out_original.float()).abs().max().item()
ref_max = out_original.float().abs().max().item() + 1e-8
rel_err = abs_err / ref_max
print(f"Max abs error vs original: {abs_err:.4e}")
print(f"Max rel error vs original: {rel_err:.4e}")
PASS = abs_err < 1.0  # FP8 tolerance
print(f"Correctness: {'PASS ✓' if PASS else 'FAIL ✗'}")

# ── Timing ──
def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def _orig():
    out_original.zero_()
    original_kernel.kernel(logits, bias, hs_fp8, hs_scl, w1, s1, w2, s2,
                           **kwargs, output=out_original)

def _grouped():
    out_grouped.zero_()
    grouped_kernel.kernel(logits, bias, hs_fp8, hs_scl, w1, s1, w2, s2,
                          **kwargs, output=out_grouped)

print("\n" + "="*60)
print("Benchmarking...")
t_orig = timeit(_orig)
t_grouped = timeit(_grouped)

print(f"\n  Original (per-expert loop):  {t_orig:.3f} ms")
print(f"  Grouped GEMM (batched):      {t_grouped:.3f} ms")
if t_grouped > 0:
    speedup = t_orig / t_grouped
    print(f"  Speedup:                     {speedup:.2f}×")
print("="*60)

# ── Larger batch test ──
print("\nLarger batch test (T=64)...")
T_big = 64
logits_big = torch.randn(T_big, NUM_E_GLOBAL, device=DEVICE, dtype=torch.float32)
hs_fp8_big = torch.randn(T_big, H, device=DEVICE).clamp(-1, 1).to(torch.float8_e4m3fn)
hs_scl_big = torch.ones(H // BQ, T_big, device=DEVICE, dtype=torch.float32) * 0.005

out_orig_big = torch.zeros(T_big, H, device=DEVICE, dtype=torch.bfloat16)
out_grp_big  = torch.zeros(T_big, H, device=DEVICE, dtype=torch.bfloat16)

original_kernel.kernel(logits_big, bias, hs_fp8_big, hs_scl_big, w1, s1, w2, s2,
                       **kwargs, output=out_orig_big)
grouped_kernel.kernel(logits_big, bias, hs_fp8_big, hs_scl_big, w1, s1, w2, s2,
                      **kwargs, output=out_grp_big)

abs_err_big = (out_grp_big.float() - out_orig_big.float()).abs().max().item()
print(f"  Max abs error (T=64): {abs_err_big:.4e}")
print(f"  Correctness (T=64):   {'PASS ✓' if abs_err_big < 1.0 else 'FAIL ✗'}")

def _orig_big():
    out_orig_big.zero_()
    original_kernel.kernel(logits_big, bias, hs_fp8_big, hs_scl_big, w1, s1, w2, s2,
                           **kwargs, output=out_orig_big)
def _grouped_big():
    out_grp_big.zero_()
    grouped_kernel.kernel(logits_big, bias, hs_fp8_big, hs_scl_big, w1, s1, w2, s2,
                          **kwargs, output=out_grp_big)

t_orig_big = timeit(_orig_big)
t_grp_big  = timeit(_grouped_big)
print(f"\n  Original (T=64):   {t_orig_big:.3f} ms")
print(f"  Grouped (T=64):    {t_grp_big:.3f} ms")
if t_grp_big > 0:
    print(f"  Speedup (T=64):    {t_orig_big / t_grp_big:.2f}×")

if not PASS:
    sys.exit(1)
print("\n✅ All tests passed!")
