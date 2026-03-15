#!/usr/bin/env python3
"""Debug version of run_local.py that skips reference implementations."""

import os
import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from safetensors.torch import load_file
from flashinfer_bench import BenchmarkConfig, Solution


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def load_single_workload(workload_uuid: str, base_dir: str):
    """Load a single workload's tensors."""
    jsonl_path = PROJECT_ROOT.parent / "mlsys26-contest/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl"

    # Find workload config
    workload_config = None
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['workload']['uuid'] == workload_uuid:
                workload_config = data['workload']
                break

    if not workload_config:
        raise ValueError(f"Workload {workload_uuid} not found")

    seq_len = workload_config['axes']['seq_len']
    expert_offset = workload_config['inputs']['local_expert_offset']['value']

    # Load tensors
    logits_file = workload_config['inputs']['routing_logits']['path'].split('/')[-1]
    bias_file = workload_config['inputs']['routing_bias']['path'].split('/')[-1]

    logits_path = os.path.join(base_dir, logits_file)
    bias_path = os.path.join(base_dir, bias_file)

    routing_logits = load_file(logits_path)['routing_logits']
    routing_bias = load_file(bias_path)['routing_bias']

    print(f"Loaded workload {workload_uuid[:8]}:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Expert offset: {expert_offset}")
    print(f"  Routing logits shape: {routing_logits.shape}")
    print(f"  Routing bias shape: {routing_bias.shape}")

    return {
        'uuid': workload_uuid,
        'seq_len': seq_len,
        'expert_offset': expert_offset,
        'routing_logits': routing_logits,
        'routing_bias': routing_bias,
        'logits_file': logits_file,
        'bias_file': bias_file
    }


def create_minimal_tensors(seq_len: int = 7, expert_offset: int = 192):
    """Create minimal test tensors for debugging."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create synthetic routing data
    num_experts = 256
    routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32, device=device)
    routing_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)

    # Create synthetic hidden states (FP8 quantized)
    hidden_size = 7168
    block_size = 128
    hidden_states = torch.randint(-127, 127, (seq_len, hidden_size), dtype=torch.int8, device=device)
    hidden_states_scale = torch.randn(hidden_size // block_size, seq_len, dtype=torch.float32, device=device)

    # Create synthetic weights (FP8 quantized)
    intermediate_size = 2048
    num_local_experts = 32

    gemm1_weights = torch.randint(-127, 127, (num_local_experts, 2 * intermediate_size, hidden_size), dtype=torch.int8, device=device)
    gemm1_weights_scale = torch.randn(num_local_experts, (2 * intermediate_size) // block_size, hidden_size // block_size, dtype=torch.float32, device=device)

    gemm2_weights = torch.randint(-127, 127, (num_local_experts, hidden_size, intermediate_size), dtype=torch.int8, device=device)
    gemm2_weights_scale = torch.randn(num_local_experts, hidden_size // block_size, intermediate_size // block_size, dtype=torch.float32, device=device)

    print(f"Created minimal synthetic tensors:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Expert offset: {expert_offset}")
    print(f"  Device: {device}")

    return {
        'seq_len': seq_len,
        'expert_offset': expert_offset,
        'routing_logits': routing_logits,
        'routing_bias': routing_bias,
        'hidden_states': hidden_states,
        'hidden_states_scale': hidden_states_scale,
        'gemm1_weights': gemm1_weights,
        'gemm1_weights_scale': gemm1_weights_scale,
        'gemm2_weights': gemm2_weights,
        'gemm2_weights_scale': gemm2_weights_scale,
    }


def run_debug_kernel(tensors: dict):
    """Run the kernel with extensive debugging."""
    device = tensors['routing_logits'].device

    # Prepare output tensor
    output = torch.zeros((tensors['seq_len'], 7168), dtype=torch.bfloat16, device=device)

    print("\n" + "="*60)
    print("RUNNING DEBUG KERNEL")
    print("="*60)

    # Import and run the kernel
    from solution.triton.local_kernel import kernel

    try:
        kernel(
            routing_logits=tensors['routing_logits'],
            routing_bias=tensors['routing_bias'],
            hidden_states=tensors['hidden_states'],
            hidden_states_scale=tensors['hidden_states_scale'],
            gemm1_weights=tensors['gemm1_weights'],
            gemm1_weights_scale=tensors['gemm1_weights_scale'],
            gemm2_weights=tensors['gemm2_weights'],
            gemm2_weights_scale=tensors['gemm2_weights_scale'],
            local_expert_offset=tensors['expert_offset'],
            routed_scaling_factor=2.5,
            output=output
        )

        print("\n✓ Kernel completed successfully!")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    except Exception as e:
        print(f"\n✗ Kernel failed with error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main debug function."""
    print("FlashInfer-MoE Local Debug Runner")
    print("=" * 40)

    # Set dataset path
    if 'FIB_DATASET_PATH' not in os.environ:
        os.environ['FIB_DATASET_PATH'] = '/home/vanshnawander/accelerated-hpc/mlsys26-contest'

    base_dir = '/home/vanshnawander/accelerated-hpc/mlsys26-contest/blob/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048'

    # Choose what to run
    print("\nChoose debug mode:")
    print("1. Use synthetic minimal tensors")
    print("2. Load real workload (specify UUID)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        # Use synthetic tensors
        seq_len = int(input("Enter sequence length (default 7): ") or "7")
        expert_offset = int(input("Enter expert offset (default 192): ") or "192")
        tensors = create_minimal_tensors(seq_len, expert_offset)

    elif choice == '2':
        # Load real workload
        workload_uuid = input("Enter workload UUID (default: b8f4f012-a32e-4356-b4e1-7665b3d598af): ").strip()
        if not workload_uuid:
            workload_uuid = "b8f4f012-a32e-4356-b4e1-7665b3d598af"

        workload = load_single_workload(workload_uuid, base_dir)
        tensors = create_minimal_tensors(workload['seq_len'], workload['expert_offset'])
        print(f"Using parameters from workload {workload_uuid}")

    else:
        print("Invalid choice")
        return

    # Run the debug kernel
    run_debug_kernel(tensors)


if __name__ == "__main__":
    main()
