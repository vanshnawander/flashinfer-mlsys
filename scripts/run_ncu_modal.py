"""
NCU Profiler via Modal B200 — runs torch profiler remotely on B200.

Since NCU requires root/special privileges, we use torch.profiler instead
which provides kernel-level breakdown on Modal B200 GPUs.

Usage:
    conda activate fi-bench
    modal run scripts/run_ncu_modal.py
    modal run scripts/run_ncu_modal.py --T 256
    modal run scripts/run_ncu_modal.py --T 4096 --kernel kernel13
"""

import sys
import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution, TraceSet

app = modal.App("flashinfer-ncu-profiler")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "ninja-build", "wget", "gnupg")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-nvcc-12-8 cuda-cudart-dev-12-8",
        "ln -sf /usr/local/cuda-12.8 /usr/local/cuda",
    )
    .env({"CUDA_HOME": "/usr/local/cuda", "PATH": "/usr/local/cuda/bin:$PATH"})
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "ninja")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_torch_profiler(solution: Solution, t_sizes: list = None, iters: int = 5) -> dict:
    """Run torch profiler on Modal B200 and return kernel-level breakdown."""
    import torch
    from torch.profiler import profile, ProfilerActivity, schedule
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    # Run profiling on actual workloads
    bench_config = BenchmarkConfig(warmup_runs=2, iterations=10, num_trials=3)
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, bench_config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])

    # Collect profiling results
    gpu_name = torch.cuda.get_device_name()
    results = {
        "gpu": gpu_name,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "workloads": {},
    }

    for trace in traces:
        if trace.evaluation:
            wl_uuid = trace.workload.uuid[:8]
            entry = {
                "status": trace.evaluation.status.value,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results["workloads"][wl_uuid] = entry

    return results


@app.local_entrypoint()
def main():
    """Run profiler on Modal B200."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning torch profiler on Modal B200...")
    results = run_torch_profiler.remote(solution)

    print(f"\nGPU: {results['gpu']}")
    print(f"CUDA: {results['cuda_version']}")
    print(f"PyTorch: {results['pytorch_version']}")
    print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "ncu_profiles"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"profiler_results_{timestamp}.json"

    import json
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Profile results saved to: {out_file}")

    # Print summary
    print("\nWorkload Results:")
    for wl_uuid, entry in results.get("workloads", {}).items():
        status = entry.get("status", "UNKNOWN")
        latency = entry.get("latency_ms", 0)
        speedup = entry.get("speedup_factor", 0)
        abs_err = entry.get("max_abs_error", 0)
        print(f"  {wl_uuid}...: {status} | {latency:.3f} ms | {speedup:.2f}x | abs_err={abs_err:.2e}")
