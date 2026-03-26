#!/bin/bash
# NCU Profiling Script for MoE Kernel on B200
#
# Usage:
#   ./scripts/ncu_profile.sh                     # Default: T=256, Triton
#   ./scripts/ncu_profile.sh --T 4096            # Large T (the failing cases)
#   ./scripts/ncu_profile.sh --T 256 --cuda      # CUDA kernel
#   ./scripts/ncu_profile.sh --T 64 --quick      # Quick metrics only
#
# Requirements:
#   - NVIDIA Nsight Compute (ncu) installed
#   - B200/A100 GPU with sufficient permissions
#   - Set: export FIB_DATASET_PATH=/path/to/traces (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs/ncu"
mkdir -p "$LOG_DIR"

# Parse args
T=256
MODE="triton"
QUICK=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --T) T="$2"; shift 2 ;;
        --cuda) MODE="cuda"; shift ;;
        --quick) QUICK=true; shift ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPORT_NAME="moe_${MODE}_T${T}_${TIMESTAMP}"

echo "=========================================="
echo "NCU Profiling: MoE Kernel"
echo "  Mode:   $MODE"
echo "  T:      $T"
echo "  Output: $LOG_DIR/$EXPORT_NAME.ncu-rep"
echo "=========================================="
echo ""

KERNEL_FLAG=""
if [ "$MODE" = "cuda" ]; then
    KERNEL_FLAG="--cuda"
fi

if [ "$QUICK" = true ]; then
    echo "Quick metrics mode — lightweight capture"
    echo ""
    
    # Key metrics for identifying bottleneck type
    ncu \
        --metrics \
            sm__throughput.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.per_cycle_active,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active,\
sm__sass_thread_inst_executed_op_fadd_pred_on.avg.pct_of_peak_sustained_active,\
sm__sass_thread_inst_executed_op_fmul_pred_on.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
launch__occupancy,\
sm__warps_active.avg.pct_of_peak_sustained_active \
        --target-processes all \
        python "$SCRIPT_DIR/ncu_profile.py" --T "$T" $KERNEL_FLAG --ncu-mode $EXTRA_ARGS

else
    echo "Full profile mode — comprehensive capture"
    echo "This may take several minutes..."
    echo ""
    
    ncu \
        --set full \
        --target-processes all \
        --export "$LOG_DIR/$EXPORT_NAME" \
        --force-overwrite \
        --cache-control all \
        --clock-control base \
        python "$SCRIPT_DIR/ncu_profile.py" --T "$T" $KERNEL_FLAG --ncu-mode $EXTRA_ARGS

    echo ""
    echo "=========================================="
    echo "Profile saved: $LOG_DIR/$EXPORT_NAME.ncu-rep"
    echo ""
    echo "To view:"
    echo "  ncu-ui $LOG_DIR/$EXPORT_NAME.ncu-rep"
    echo ""
    echo "To view specific kernel:"
    echo "  ncu --import $LOG_DIR/$EXPORT_NAME.ncu-rep --page details"
    echo "=========================================="
fi

echo ""
echo "Key metrics to check:"
echo "  1. sm__throughput (compute utilization)"
echo "  2. dram__throughput (memory bandwidth utilization)"
echo "  3. sm__warps_active (occupancy)"
echo "  4. sm__inst_executed_pipe_tensor (tensor core usage)"
echo ""
echo "If dram >> sm → memory-bound → optimize data layout, reduce traffic"
echo "If sm >> dram → compute-bound → optimize tile sizes, use FP8"
echo "If both low → latency-bound → reduce kernel launches, increase parallelism"
