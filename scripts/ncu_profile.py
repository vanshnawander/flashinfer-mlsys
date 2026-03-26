#!/usr/bin/env python3
"""
NCU-compatible profiling wrapper for MoE kernel.

Usage:
    # Direct Python run (captures torch profiler):
    python scripts/ncu_profile.py --T 256

    # With NVIDIA Nsight Compute (on B200/A100):
    ncu --set full --target-processes all \
        --export profile_T256 \
        python scripts/ncu_profile.py --T 256 --ncu-mode

    # Quick metrics only (no full trace):
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.per_cycle_active,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active \
        python scripts/ncu_profile.py --T 256 --ncu-mode

    # Profiling specific kernels:
    ncu --kernel-name "_fused_gemm1_swiglu_kernel" --launch-count 3 \
        python scripts/ncu_profile.py --T 256 --ncu-mode

Notes:
    - When --ncu-mode is set, warmup is skipped and only 1 iteration runs
    - NCU captures ALL GPU kernel launches; use --kernel-name to filter
    - Export format: --export creates .ncu-rep file viewable in Nsight Compute GUI
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
    E = 32

    fp8_dtype = getattr(torch, 'float8_e4m3fn', torch.int8)

    if fp8_dtype == torch.int8:
        hidden_states = torch.randint(-127, 127, (T, H), dtype=torch.int8, device=device)
    else:
        hidden_states = torch.randn(T, H, device=device, dtype=torch.float32).to(fp8_dtype)

    hidden_states_scale = torch.rand(H // BQ, T, dtype=torch.float32, device=device) * 0.1 + 0.01

    routing_logits = torch.randn(T, 256, dtype=torch.float32, device=device)
    routing_bias = torch.randn(256, dtype=torch.bfloat16, device=device) * 0.01

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


def run_with_torch_profiler(kernel_fn, tensors, iters=5):
    """Run with PyTorch profiler (works on any GPU)."""
    from torch.profiler import profile, ProfilerActivity, schedule

    def run_once():
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

    # Warmup
    for _ in range(3):
        run_once()
        torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=iters, repeat=1),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(2 + iters):
            run_once()
            torch.cuda.synchronize()
            prof.step()

    # Print summary
    print("\n" + "=" * 80)
    print("TORCH PROFILER — CUDA Kernel Summary (sorted by CUDA time)")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
        max_name_column_width=60,
    ))

    # Export Chrome trace
    trace_path = str(PROJECT_ROOT / "logs" / "torch_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    print("Open in chrome://tracing or https://ui.perfetto.dev/")

    return prof


def run_for_ncu(kernel_fn, tensors):
    """Single run for NCU capture (no warmup, single iteration)."""
    print("NCU mode: running single iteration for capture...")

    # Force compilation first (Triton JIT)
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

    # The actual profiled run
    print("Starting profiled run...")
    torch.cuda.cudart().cudaProfilerStart()

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

    torch.cuda.cudart().cudaProfilerStop()
    print("NCU capture complete.")


def run_manual_timing(kernel_fn, tensors, warmup=5, iters=20):
    """Manual CUDA event timing with per-section breakdown."""
    device = tensors['output'].device

    def run_once():
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

    # Warmup
    for _ in range(warmup):
        run_once()
        torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start_event.record()
        run_once()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times.sort()
    print(f"\n{'='*60}")
    print(f"CUDA Event Timing ({iters} iterations)")
    print(f"{'='*60}")
    print(f"  Min:    {times[0]:.3f} ms")
    print(f"  P25:    {times[len(times)//4]:.3f} ms")
    print(f"  Median: {times[len(times)//2]:.3f} ms")
    print(f"  P75:    {times[3*len(times)//4]:.3f} ms")
    print(f"  Max:    {times[-1]:.3f} ms")
    print(f"  Mean:   {sum(times)/len(times):.3f} ms")

    return times


def main():
    parser = argparse.ArgumentParser(description='MoE kernel profiling')
    parser.add_argument('--T', type=int, default=256, help='Sequence length')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA binding')
    parser.add_argument('--ncu-mode', action='store_true',
                        help='NCU mode: single run, no warmup')
    parser.add_argument('--torch-profile', action='store_true',
                        help='Use torch.profiler (detailed kernel breakdown)')
    parser.add_argument('--expert-offset', type=int, default=192)
    parser.add_argument('--iters', type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name()
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Config: T={args.T}, expert_offset={args.expert_offset}")
    print()

    # Create data
    tensors = create_synthetic_tensors(args.T, args.expert_offset)

    # Load kernel
    if args.cuda:
        from solution.cuda.binding import kernel as kernel_fn
        print("Loaded CUDA kernel")
    else:
        from solution.triton.kernel import kernel as kernel_fn
        print("Loaded Triton kernel")

    if args.ncu_mode:
        run_for_ncu(kernel_fn, tensors)
    elif args.torch_profile:
        run_with_torch_profiler(kernel_fn, tensors, args.iters)
    else:
        run_manual_timing(kernel_fn, tensors, iters=args.iters)


if __name__ == '__main__':
    main()
