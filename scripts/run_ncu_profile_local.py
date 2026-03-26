"""
NCU Profiler — uses flashinfer_bench built-in NCU API.

Usage:
    conda activate fi-bench
    python scripts/run_ncu_profile.py                    # Profile all workloads
    python scripts/run_ncu_profile.py --workload-idx 0   # Profile specific workload

Output saved to: ncu_profiles/<solution_name>_<timestamp>/

Requires local CUDA GPU + FIB_DATASET_PATH env var.
Uses flashinfer_bench.agents.flashinfer_bench_run_ncu() per starter-kit README.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_ncu_profile(workload_idx: int = None, ncu_set: str = "detailed", page: str = "details"):
    """Run NCU profiling using the flashinfer-bench API."""
    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    # Load solution
    solution_path = PROJECT_ROOT / "solution.json"
    if not solution_path.exists():
        print("No solution.json found. Run: python scripts/pack_solution.py")
        return

    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Solution: {solution.name} (language: {solution.spec.language})")

    # Load trace set
    fib_path = os.environ.get("FIB_DATASET_PATH")
    if not fib_path:
        print("ERROR: Set FIB_DATASET_PATH environment variable")
        return

    trace_set = TraceSet.from_path(fib_path)
    definition = trace_set.definitions.get(solution.definition)
    if not definition:
        print(f"Definition '{solution.definition}' not in trace set")
        return

    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        print("No workloads found")
        return

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "ncu_profiles" / f"{solution.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select workloads to profile
    if workload_idx is not None:
        if workload_idx >= len(workloads):
            print(f"Workload index {workload_idx} out of range (0-{len(workloads)-1})")
            return
        selected = [(workload_idx, workloads[workload_idx])]
    else:
        # Profile first 3 workloads by default (small, medium, large)
        indices = [0, len(workloads) // 2, len(workloads) - 1]
        selected = [(i, workloads[i]) for i in indices if i < len(workloads)]

    print(f"Profiling {len(selected)} workload(s) with set='{ncu_set}', page='{page}'")
    print(f"Output: {out_dir}\n")

    for idx, workload in selected:
        print(f"--- Workload {idx}: {workload.uuid[:12]}... ---")

        try:
            output = flashinfer_bench_run_ncu(
                solution=solution,
                workload=workload,
                set=ncu_set,
                page=page,
                timeout=300,
            )

            # Save output
            out_file = out_dir / f"workload_{idx}_{workload.uuid[:8]}.txt"
            with open(out_file, "w") as f:
                f.write(f"# NCU Profile: {solution.name}\n")
                f.write(f"# Workload: {workload.uuid}\n")
                f.write(f"# Set: {ncu_set}, Page: {page}\n")
                f.write(f"# Timestamp: {datetime.now().isoformat()}\n\n")
                f.write(str(output))

            print(f"  Saved: {out_file}")

        except Exception as e:
            print(f"  ERROR: {e}")
            # Save error too
            err_file = out_dir / f"workload_{idx}_{workload.uuid[:8]}_error.txt"
            with open(err_file, "w") as f:
                f.write(f"Error profiling workload {workload.uuid}:\n{e}\n")

    print(f"\nDone! NCU profiles saved to: {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NCU Profiler for MoE kernel")
    parser.add_argument("--workload-idx", type=int, default=None, help="Profile specific workload index")
    parser.add_argument("--set", default="detailed", choices=["basic", "detailed", "full"], help="NCU metric set")
    parser.add_argument("--page", default="details", choices=["details", "source", "raw"], help="NCU page")
    args = parser.parse_args()

    run_ncu_profile(workload_idx=args.workload_idx, ncu_set=args.set, page=args.page)
