#!/usr/bin/env python3
"""
Local debug runner for MoE kernels.

Supports both Triton and CUDA solutions with synthetic or real workload data.
Usage:
    python scripts/run_local_debug.py                  # Triton, synthetic, T=64
    python scripts/run_local_debug.py --cuda            # CUDA binding
    python scripts/run_local_debug.py --T 4096          # Large T test
    python scripts/run_local_debug.py --T 7 --verbose   # Debug prints
"""

import os
import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def create_synthetic_tensors(T: int, expert_offset: int = 192, device: str = 'cuda'):
    """Create synthetic test tensors matching the benchmark format."""
    H = 7168
    I = 2048
    BQ = 128
    E = 32  # local experts

    # FP8 = torch.float8_e4m3fn if available, else int8 as raw bytes
    fp8_dtype = getattr(torch, 'float8_e4m3fn', torch.int8)

    # Hidden states (FP8)
    if fp8_dtype == torch.int8:
        hidden_states = torch.randint(-127, 127, (T, H), dtype=torch.int8, device=device)
    else:
        hidden_states = torch.randn(T, H, device=device, dtype=torch.float32).to(fp8_dtype)

    hidden_states_scale = torch.rand(H // BQ, T, dtype=torch.float32, device=device) * 0.1 + 0.01

    # Routing
    routing_logits = torch.randn(T, 256, dtype=torch.float32, device=device)
    routing_bias = torch.randn(256, dtype=torch.bfloat16, device=device) * 0.01

    # Weights (FP8)
    if fp8_dtype == torch.int8:
        gemm1_weights = torch.randint(-127, 127, (E, 2 * I, H), dtype=torch.int8, device=device)
        gemm2_weights = torch.randint(-127, 127, (E, H, I), dtype=torch.int8, device=device)
    else:
        gemm1_weights = torch.randn(E, 2 * I, H, device=device, dtype=torch.float32).to(fp8_dtype)
        gemm2_weights = torch.randn(E, H, I, device=device, dtype=torch.float32).to(fp8_dtype)

    gemm1_weights_scale = torch.rand(E, (2 * I) // BQ, H // BQ, dtype=torch.float32, device=device) * 0.1 + 0.01
    gemm2_weights_scale = torch.rand(E, H // BQ, I // BQ, dtype=torch.float32, device=device) * 0.1 + 0.01

    output = torch.zeros(T, H, dtype=torch.bfloat16, device=device)

    return {
        'routing_logits': routing_logits,
        'routing_bias': routing_bias,
        'hidden_states': hidden_states,
        'hidden_states_scale': hidden_states_scale,
        'gemm1_weights': gemm1_weights,
        'gemm1_weights_scale': gemm1_weights_scale,
        'gemm2_weights': gemm2_weights,
        'gemm2_weights_scale': gemm2_weights_scale,
        'local_expert_offset': expert_offset,
        'routed_scaling_factor': 2.5,
        'output': output,
    }


def run_kernel(kernel_fn, tensors, warmup=3, benchmark_iters=10):
    """Run kernel with warmup and timing."""
    device = tensors['output'].device

    # Warmup
    for _ in range(warmup):
        tensors['output'].zero_()
        kernel_fn(
            routing_logits=tensors['routing_logits'],
            routing_bias=tensors['routing_bias'],
            hidden_states=tensors['hidden_states'],
            hidden_states_scale=tensors['hidden_states_scale'],
            gemm1_weights=tensors['gemm1_weights'],
            gemm1_weights_scale=tensors['gemm1_weights_scale'],
            gemm2_weights=tensors['gemm2_weights'],
            gemm2_weights_scale=tensors['gemm2_weights_scale'],
            local_expert_offset=tensors['local_expert_offset'],
            routed_scaling_factor=tensors['routed_scaling_factor'],
            output=tensors['output'],
        )
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(benchmark_iters):
        tensors['output'].zero_()
        torch.cuda.synchronize()

        start = time.perf_counter()
        kernel_fn(
            routing_logits=tensors['routing_logits'],
            routing_bias=tensors['routing_bias'],
            hidden_states=tensors['hidden_states'],
            hidden_states_scale=tensors['hidden_states_scale'],
            gemm1_weights=tensors['gemm1_weights'],
            gemm1_weights_scale=tensors['gemm1_weights_scale'],
            gemm2_weights=tensors['gemm2_weights'],
            gemm2_weights_scale=tensors['gemm2_weights_scale'],
            local_expert_offset=tensors['local_expert_offset'],
            routed_scaling_factor=tensors['routed_scaling_factor'],
            output=tensors['output'],
        )
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return times


def main():
    parser = argparse.ArgumentParser(description='Local MoE kernel debug runner')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA binding instead of Triton')
    parser.add_argument('--T', type=int, default=64, help='Sequence length (default: 64)')
    parser.add_argument('--expert-offset', type=int, default=192, help='Expert offset (default: 192)')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=10, help='Benchmark iterations')
    parser.add_argument('--verbose', action='store_true', help='Print debug info')
    parser.add_argument('--compare', action='store_true', help='Compare Triton vs CUDA')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Create synthetic data
    print(f"Creating synthetic tensors: T={args.T}, expert_offset={args.expert_offset}")
    tensors = create_synthetic_tensors(args.T, args.expert_offset, device)
    print(f"  hidden_states: {tensors['hidden_states'].shape} ({tensors['hidden_states'].dtype})")
    print(f"  gemm1_weights: {tensors['gemm1_weights'].shape} ({tensors['gemm1_weights'].dtype})")
    print()

    if args.compare:
        # Run both and compare
        print("=" * 60)
        print("COMPARING TRITON vs CUDA")
        print("=" * 60)

        from solution.triton.kernel import kernel as triton_kernel
        from solution.cuda.binding import kernel as cuda_kernel

        # Triton
        print("\n--- Triton ---")
        try:
            triton_times = run_kernel(triton_kernel, tensors, args.warmup, args.iters)
            triton_out = tensors['output'].clone()
            print(f"  Median: {sorted(triton_times)[len(triton_times)//2]:.3f} ms")
            print(f"  Output range: [{triton_out.float().min():.4f}, {triton_out.float().max():.4f}]")
        except Exception as e:
            print(f"  FAILED: {e}")
            triton_out = None

        # CUDA
        print("\n--- CUDA ---")
        try:
            cuda_times = run_kernel(cuda_kernel, tensors, args.warmup, args.iters)
            cuda_out = tensors['output'].clone()
            print(f"  Median: {sorted(cuda_times)[len(cuda_times)//2]:.3f} ms")
            print(f"  Output range: [{cuda_out.float().min():.4f}, {cuda_out.float().max():.4f}]")
        except Exception as e:
            print(f"  FAILED: {e}")
            cuda_out = None

        # Compare outputs
        if triton_out is not None and cuda_out is not None:
            diff = (triton_out.float() - cuda_out.float()).abs()
            print(f"\n--- Comparison ---")
            print(f"  Max abs diff: {diff.max():.6e}")
            print(f"  Mean abs diff: {diff.mean():.6e}")

    else:
        # Single kernel run
        if args.cuda:
            print("Loading CUDA kernel...")
            from solution.cuda.binding import kernel as kernel_fn
            kernel_name = "CUDA"
        else:
            print("Loading Triton kernel...")
            from solution.triton.kernel import kernel as kernel_fn
            kernel_name = "Triton"

        print(f"\nRunning {kernel_name} kernel (T={args.T})...")
        try:
            times = run_kernel(kernel_fn, tensors, args.warmup, args.iters)
            out = tensors['output']

            print(f"\n✓ {kernel_name} kernel completed successfully!")
            print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
            print(f"  Output range: [{out.float().min():.4f}, {out.float().max():.4f}]")
            print(f"  Non-zero elements: {(out != 0).sum().item()} / {out.numel()}")
            print(f"\n  Timing ({args.iters} iters):")
            print(f"    Min:    {min(times):.3f} ms")
            print(f"    Median: {sorted(times)[len(times)//2]:.3f} ms")
            print(f"    Max:    {max(times):.3f} ms")

            if args.verbose:
                print(f"\n  All times: {[f'{t:.3f}' for t in times]}")
                print(f"  Output sample (first 5 tokens, first 10 cols):")
                print(out[:5, :10].float())

        except Exception as e:
            print(f"\n✗ {kernel_name} kernel FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
