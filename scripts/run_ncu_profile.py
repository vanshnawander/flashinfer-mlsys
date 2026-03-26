"""
NCU Profiler on Modal B200 — uses flashinfer_bench built-in NCU API.

Usage:
    conda activate fi-bench
    modal run scripts/run_ncu_profile.py                     # Profile 3 representative workloads
    modal run scripts/run_ncu_profile.py --workload-idx 5    # Profile specific workload

Output saved to: ncu_profiles/<solution_name>_<timestamp>/
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution, TraceSet

app = modal.App("flashinfer-ncu")

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
def run_ncu_on_b200(solution: Solution, workload_idx: int, ncu_set: str, page: str) -> dict:
    """Run NCU profiling on B200 for a single workload."""
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    workloads = trace_set.workloads.get(solution.definition, [])

    if workload_idx >= len(workloads):
        return {"error": "workload_idx out of range", "max": len(workloads) - 1}

    workload = workloads[workload_idx]
    print("Profiling workload", workload_idx, "uuid:", workload.uuid[:12])

    try:
        output = flashinfer_bench_run_ncu(
            solution=solution,
            workload=workload,
            set=ncu_set,
            page=page,
            timeout=300,
        )
        return {
            "workload_idx": workload_idx,
            "uuid": workload.uuid,
            "ncu_output": str(output),
        }
    except Exception as e:
        return {
            "workload_idx": workload_idx,
            "uuid": workload.uuid,
            "error": str(e),
        }


@app.local_entrypoint()
def main(
    workload_idx: int = -1,
    ncu_set: str = "detailed",
    page: str = "details",
):
    """Pack solution and run NCU on Modal B200."""
    from scripts.pack_solution import pack_solution

    print("Packing solution...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())
    print("Solution:", solution.name, "| Language:", solution.spec.language)

    # Determine which workloads to profile
    if workload_idx >= 0:
        indices = [workload_idx]
    else:
        # Profile small (0), medium (mid), large (last)
        indices = [0, 9, 18]  # Adjust based on dataset

    # Create output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "ncu_profiles" / (solution.name + "_" + timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Output:", out_dir)
    print("Profiling", len(indices), "workload(s) on B200...\n")

    for idx in indices:
        print(f"--- Workload {idx} ---")
        result = run_ncu_on_b200.remote(solution, idx, ncu_set, page)

        if "error" in result:
            print("  ERROR:", result["error"])
            out_file = out_dir / f"workload_{idx}_error.txt"
            with open(out_file, "w") as f:
                f.write(str(result))
        else:
            out_file = out_dir / f"workload_{idx}_{result['uuid'][:8]}.txt"
            with open(out_file, "w") as f:
                f.write(f"# NCU Profile: {solution.name}\n")
                f.write(f"# Workload idx={idx}, uuid={result['uuid']}\n")
                f.write(f"# Set: {ncu_set}, Page: {page}\n\n")
                f.write(result["ncu_output"])
            print("  Saved:", out_file.name)
            # Print first 50 lines as preview
            lines = result["ncu_output"].split("\n")
            for line in lines[:50]:
                print("  ", line)
            if len(lines) > 50:
                print(f"  ... ({len(lines) - 50} more lines in file)")

    print(f"\nDone! Profiles in: {out_dir}")
