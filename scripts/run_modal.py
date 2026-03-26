"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Changes from starter-kit template:
    - Image: Added CUDA toolkit (nvcc, build-essential, ninja) for torch binding CUDA solutions
    - print_results: Enhanced with file logging and workload parameter loading
    - debug_env: Optional helper to check B200 build environment (use --debug flag)
"""

import sys
import json
import os
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Image with CUDA toolkit for torch-binding CUDA solutions.
# Starter-kit default only has: debian_slim + pip_install(flashinfer-bench, torch, triton, numpy)
# We add: build-essential, ninja, nvcc (cuda-nvcc-12-8) for JIT-compiling .cu files.
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
def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def load_workload_parameters(workload_uuid: str, definition_name: str) -> dict:
    """Load all parameters for a specific workload from JSONL file."""
    jsonl_path = PROJECT_ROOT.parent / "mlsys26-contest/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl"
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['workload']['uuid'] == workload_uuid:
                    workload = data['workload']
                    return {
                        'seq_len': workload['axes']['seq_len'],
                        'local_expert_offset': workload['inputs']['local_expert_offset']['value'],
                        'routed_scaling_factor': workload['inputs']['routed_scaling_factor']['value'],
                    }
    except Exception:
        pass
    return {}


def print_results(results: dict):
    """Print benchmark results in a formatted way and save to log file."""
    # Create logs directory
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"benchmark_results_{timestamp}.log"

    print(f"\nSaving results to: {log_file}")

    log_lines = [
        "=" * 80,
        f"BENCHMARK RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80, "",
    ]

    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        log_lines.append(f"Definition: {def_name}")
        log_lines.append("-" * 40)

        for workload_uuid, result in traces.items():
            wp = load_workload_parameters(workload_uuid, def_name)
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            log_lines.append(f"Workload UUID: {workload_uuid}")
            log_lines.append(f"  Status: {status}")
            if wp:
                log_lines.append(f"  Seq Len: {wp.get('seq_len', 'N/A')}")

            if result.get("latency_ms") is not None:
                latency = result["latency_ms"]
                print(f" | {latency:.3f} ms", end="")
                log_lines.append(f"  Latency: {latency:.3f} ms")

            if result.get("speedup_factor") is not None:
                speedup = result["speedup_factor"]
                print(f" | {speedup:.2f}x speedup", end="")
                log_lines.append(f"  Speedup: {speedup:.2f}x")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")
                log_lines.append(f"  Max Abs Error: {abs_err:.2e}")
                log_lines.append(f"  Max Rel Error: {rel_err:.2e}")

            if result.get("reference_latency_ms") is not None:
                log_lines.append(f"  Ref Latency: {result['reference_latency_ms']:.3f} ms")

            log_lines.append(f"  Solution: {result.get('solution', 'N/A')}")
            log_lines.append("")
            print()

    try:
        with open(log_file, 'w') as f:
            f.write('\n'.join(log_lines) + '\n')
        print(f"Results saved to {log_file}")
    except Exception as e:
        print(f"Error saving log: {e}")

    # Summary
    total = sum(len(traces) for traces in results.values())
    passed = sum(
        1 for traces in results.values()
        for r in traces.values()
        if r.get("status") == "PASSED"
    )
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed}/{total} PASSED")
    print(f"Log: {log_file}")
    print(f"{'=' * 60}")


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
