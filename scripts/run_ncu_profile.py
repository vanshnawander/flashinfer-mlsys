"""
NCU Profiler on Modal B200.

Usage:
    conda activate fi-bench
    modal run scripts/run_ncu_profile.py
    modal run scripts/run_ncu_profile.py --workload-idx 5
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

# Need NVIDIA repo for nsight-compute (ncu). NOT installing old cuda-nvcc.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "ninja-build", "wget", "gnupg")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y nsight-compute-2026.1.0",
        "ln -sf /opt/nvidia/nsight-compute/2026.1.0/ncu /usr/local/bin/ncu",
    )
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "ninja")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_ncu_on_b200(solution: Solution, workload_idx: int, ncu_set: str, page: str) -> dict:
    """Run NCU profiling on B200."""
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    traces = trace_set.workloads.get(solution.definition, [])

    if workload_idx >= len(traces):
        return {"error": f"idx {workload_idx} out of range (max {len(traces)-1})"}

    trace_obj = traces[workload_idx]
    wl = trace_obj.workload if hasattr(trace_obj, 'workload') else trace_obj
    wl_id = wl.uuid[:12] if hasattr(wl, 'uuid') else str(workload_idx)

    try:
        output = flashinfer_bench_run_ncu(
            solution=solution,
            workload=wl,
            trace_set_path=TRACE_SET_PATH,
            set=ncu_set,
            page=page,
            kernel_name=".*",
            timeout=300,
        )
        return {"workload_idx": workload_idx, "wl_id": wl_id, "ncu_output": str(output)}
    except Exception as e:
        import traceback
        return {"workload_idx": workload_idx, "wl_id": wl_id,
                "error": str(e), "traceback": traceback.format_exc()[:1500]}


@app.local_entrypoint()
def main(workload_idx: int = -1, ncu_set: str = "detailed", page: str = "details"):
    from scripts.pack_solution import pack_solution

    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Solution: {solution.name}")

    indices = [workload_idx] if workload_idx >= 0 else [0, 9, 18]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "ncu_profiles" / f"{solution.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        print(f"\n--- Workload {idx} ---")
        result = run_ncu_on_b200.remote(solution, idx, ncu_set, page)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            (out_dir / f"workload_{idx}_error.txt").write_text(str(result))
        else:
            wl_id = result.get("wl_id", str(idx))
            out_file = out_dir / f"workload_{idx}_{wl_id}.txt"
            out_file.write_text(result["ncu_output"])
            print(f"  Saved: {out_file.name}")
            for line in result["ncu_output"].split("\n")[:80]:
                print(f"  {line}")

    print(f"\nDone! Profiles in: {out_dir}")
