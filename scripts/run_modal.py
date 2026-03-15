"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
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

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
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
    # Go up one level from flashinfer-mlsys to find mlsys26-contest
    jsonl_path = PROJECT_ROOT.parent / "mlsys26-contest/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl"
    
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['workload']['uuid'] == workload_uuid:
                    workload = data['workload']
                    return {
                        'uuid': workload_uuid,
                        'seq_len': workload['axes']['seq_len'],
                        'local_expert_offset': workload['inputs']['local_expert_offset']['value'],
                        'routed_scaling_factor': workload['inputs']['routed_scaling_factor']['value'],
                        'routing_logits_path': workload['inputs']['routing_logits']['path'],
                        'routing_bias_path': workload['inputs']['routing_bias']['path'],
                        'routing_logits_tensor_key': workload['inputs']['routing_logits']['tensor_key'],
                        'routing_bias_tensor_key': workload['inputs']['routing_bias']['tensor_key'],
                        'hidden_states_type': workload['inputs']['hidden_states']['type'],
                        'hidden_states_scale_type': workload['inputs']['hidden_states_scale']['type'],
                        'gemm1_weights_type': workload['inputs']['gemm1_weights']['type'],
                        'gemm1_weights_scale_type': workload['inputs']['gemm1_weights_scale']['type'],
                        'gemm2_weights_type': workload['inputs']['gemm2_weights']['type'],
                        'gemm2_weights_scale_type': workload['inputs']['gemm2_weights_scale']['type']
                    }
    except Exception as e:
        print(f"Error loading workload parameters: {e}")
        return {}
    
    return {}


def print_results(results: dict):
    """Print benchmark results and save to log file with all parameters."""
    # Create logs directory if it doesn't exist
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"benchmark_results_{timestamp}.log"
    
    print(f"\nSaving results to: {log_file}")
    
    # Prepare log content
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append(f"BENCHMARK RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 80)
    log_lines.append("")
    
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        log_lines.append(f"Definition: {def_name}")
        log_lines.append("-" * 40)
        
        for workload_uuid, result in traces.items():
            # Load all workload parameters
            workload_params = load_workload_parameters(workload_uuid, def_name)
            
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")
            log_lines.append(f"Workload UUID: {workload_uuid}")
            log_lines.append(f"  Status: {status}")
            
            # Log all workload parameters
            if workload_params:
                log_lines.append("  Workload Parameters:")
                log_lines.append(f"    Sequence Length: {workload_params.get('seq_len', 'N/A')}")
                log_lines.append(f"    Local Expert Offset: {workload_params.get('local_expert_offset', 'N/A')}")
                log_lines.append(f"    Routed Scaling Factor: {workload_params.get('routed_scaling_factor', 'N/A')}")
                log_lines.append(f"    Routing Logits Path: {workload_params.get('routing_logits_path', 'N/A')}")
                log_lines.append(f"    Routing Bias Path: {workload_params.get('routing_bias_path', 'N/A')}")
                log_lines.append(f"    Routing Logits Tensor Key: {workload_params.get('routing_logits_tensor_key', 'N/A')}")
                log_lines.append(f"    Routing Bias Tensor Key: {workload_params.get('routing_bias_tensor_key', 'N/A')}")
                log_lines.append(f"    Hidden States Type: {workload_params.get('hidden_states_type', 'N/A')}")
                log_lines.append(f"    Hidden States Scale Type: {workload_params.get('hidden_states_scale_type', 'N/A')}")
                log_lines.append(f"    GEMM1 Weights Type: {workload_params.get('gemm1_weights_type', 'N/A')}")
                log_lines.append(f"    GEMM1 Weights Scale Type: {workload_params.get('gemm1_weights_scale_type', 'N/A')}")
                log_lines.append(f"    GEMM2 Weights Type: {workload_params.get('gemm2_weights_type', 'N/A')}")
                log_lines.append(f"    GEMM2 Weights Scale Type: {workload_params.get('gemm2_weights_scale_type', 'N/A')}")
            
            # Log performance results
            log_lines.append("  Performance Results:")
            if result.get("latency_ms") is not None:
                latency = result['latency_ms']
                print(f" | {latency:.3f} ms", end="")
                log_lines.append(f"    Latency: {latency:.3f} ms")

            if result.get("speedup_factor") is not None:
                speedup = result['speedup_factor']
                print(f" | {speedup:.2f}x speedup", end="")
                log_lines.append(f"    Speedup: {speedup:.2f}x")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")
                log_lines.append(f"    Max Absolute Error: {abs_err:.2e}")
                log_lines.append(f"    Max Relative Error: {rel_err:.2e}")
            
            if result.get("reference_latency_ms") is not None:
                ref_latency = result['reference_latency_ms']
                log_lines.append(f"    Reference Latency: {ref_latency:.3f} ms")
            
            log_lines.append(f"    Solution: {result.get('solution', 'N/A')}")
            log_lines.append("")
            print()

    # Write to log file in append mode
    try:
        with open(log_file, 'a') as f:
            f.write('\n'.join(log_lines) + '\n')
        print(f"Results successfully saved to {log_file}")
    except Exception as e:
        print(f"Error saving log file: {e}")
    
    # Also print summary to console
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    total_workloads = sum(len(traces) for traces in results.values())
    successful_workloads = sum(
        1 for traces in results.values() 
        for result in traces.values() 
        if result.get("status") == "success"
    )
    print(f"Total workloads: {total_workloads}")
    print(f"Successful: {successful_workloads}")
    print(f"Failed: {total_workloads - successful_workloads}")
    print(f"Log file: {log_file}")
    print("=" * 60)


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
