# MoE Reference Breakdown and Translation Roadmap (Triton + CUDA)

## 1) What this reference really implements

This function is one fused MoE layer for DeepSeek-style settings, not a full model:

- Inputs: routing logits/bias, FP8 hidden, FP8 expert weights, block scales, local expert offset
- Core math:
  1. FP8 block dequantize (hidden + weights)
  2. Routing (sigmoid + grouped topk + topk experts)
  3. Local expert compute: GEMM1 -> SwiGLU -> GEMM2
  4. Weighted accumulation into output
- Output: `bfloat16[T, H]`

Fixed geometry from definition:

- `E_global=256`, `E_local=32`, `H=7168`, `I=2048`, `TOP_K=8`
- Grouped routing: `N_GROUP=8`, `TOPK_GROUP=4`, group size `32`
- Block quantization size = `128`

## 2) Reference math by stage

## Stage A: FP8 block-scale dequant

Hidden:

- `hidden_states`: `[T, H]` FP8
- `hidden_states_scale`: `[H/128, T]` FP32
- Expand scale over block of 128 in hidden dim and multiply

Weights:

- `gemm1_weights`: `[E_local, 2I, H]` FP8
- `gemm1_weights_scale`: `[E_local, (2I)/128, H/128]` FP32
- `gemm2_weights`: `[E_local, H, I]` FP8
- `gemm2_weights_scale`: `[E_local, H/128, I/128]` FP32

Reference expands full dequant tensors, but optimized kernels should dequantize on-the-fly per tile.

## Stage B: DeepSeek no-aux routing

Per token:

1. `s = sigmoid(logits)`
2. `s_with_bias = s + bias`
3. Reshape to `[8 groups, 32 experts]`
4. Group score = sum of top-2 in each group
5. Keep top-4 groups
6. Within kept groups, pick top-8 experts globally
7. Combine weights from `s` (not `s_with_bias`), normalize sum to 1, multiply by `routed_scaling_factor`

Outputs of routing stage needed by compute stage:

- `topk_idx[T, 8]` (global expert ids)
- `topk_w[T, 8]` (normalized route weights)

## Stage C: Local expert compute

For each local expert `le` with global id `ge = local_expert_offset + le`:

1. Gather tokens where `ge` is in token top-8
2. GEMM1: `[Tk, H] x [H, 2I] -> [Tk, 2I]`
3. SwiGLU: split to two `[Tk, I]` halves, `C = silu(X2) * X1`
4. GEMM2: `[Tk, I] x [I, H] -> [Tk, H]`
5. Scale by token route weight for that expert, accumulate into output token rows

## 3) GPU translation strategy (important)

Do not implement as one monolithic kernel first. Use staged kernels with buffers:

1. `routing_kernel` -> writes `topk_idx`, `topk_w`
2. `dispatch_kernel` -> builds token list per local expert
3. `expert_ffn_kernel` -> grouped expert GEMMs + SwiGLU + accumulation
4. Optional fuse dispatch+compute later

Reason: correctness and debug are much easier, then you can fuse hot paths after parity.

## 4) Triton implementation roadmap

## Triton phase 1 (correct baseline)

- Kernel R (routing):
  - Program per token
  - Load 256 logits + bias, sigmoid in FP32
  - Group top2/top4/top8 using register/local arrays
  - Store `topk_idx[T,8]`, `topk_w[T,8]`

- Kernel D (dispatch):
  - Build compact `(expert_local, token_id, weight)` arrays
  - Prefer prefix-sum based packing to avoid atomics bottleneck

- Kernel F1/F2:
  - F1: tiled GEMM `A_e x W13_e^T`, on-the-fly dequant
  - SwiGLU in FP32 or mixed FP32 accumulate
  - F2: tiled GEMM `C x W2_e^T`, on-the-fly dequant
  - Multiply by route weight and scatter-add to output

## Triton phase 2 (performance)

- Fuse F1 + SwiGLU + F2 with persistent blocks
- Reorder tokens by expert for contiguous memory
- Tune block sizes for Blackwell (e.g. larger K tiles if SRAM allows)
- Keep accumulators in FP32, cast final output to BF16

## 5) CUDA implementation roadmap

## CUDA phase 1 (correct baseline)

- `binding.py` calls C++ launcher with exact tensor order
- CUDA kernels:
  - `routing.cu`: one block/token or warp/token, compute topk indices/weights
  - `dispatch.cu`: create per-expert token index lists + counts
  - `moe_ffn.cu`: expert GEMM1, SwiGLU, GEMM2, weighted accumulation

For baseline speed, you can use CUTLASS/CUBLAS for GEMMs after dispatch, then custom kernel for weighted scatter-add.

## CUDA phase 2 (high performance)

- Custom fused kernel per expert tile
- On-the-fly FP8 dequant in shared memory tiles
- Use warp specialization:
  - warps for global load/dequant
  - warps for MMA compute
  - warps for epilogue/scatter
- Minimize global atomics by assigning unique token tiles when possible

## 6) FP8-specific implementation notes

- Treat storage dtype exactly as `float8_e4m3fn`
- Always convert FP8 inputs to FP16/FP32 for arithmetic
- Multiply with block scale before MMA/accumulate
- Use FP32 accumulation for:
  - routing sigmoid/topk logic
  - GEMM accumulators (at least epilogue path)
  - output accumulation across experts
- Cast to BF16 only at final output write

Block-scale indexing formulas (critical):

- Hidden scale index: `scale_h = hidden_states_scale[h_block, token]`
  - `h_block = h // 128`
- GEMM1 scale index: `s13 = gemm1_weights_scale[e, out_block, h_block]`
  - `out_block = out_col // 128`, `h_block = h // 128`
- GEMM2 scale index: `s2 = gemm2_weights_scale[e, h_block, i_block]`
  - `h_block = h // 128`, `i_block = i // 128`

## 7) GPU internals checklist (what decides performance)

- Occupancy vs register pressure
  - Routing kernels: keep per-token temp arrays small
  - GEMM kernels: do not over-unroll into register spill

- Memory hierarchy
  - Coalesce reads for hidden and weights
  - Stage dequant scales in shared memory when reused
  - Avoid materializing full dequantized W13/W2 in global memory

- Warp-level operations
  - Use warp reductions/shuffles for topk/group scoring
  - Use cooperative groups for block-level expert dispatch

- Atomic contention
  - Output accumulation can bottleneck if many experts hit same token
  - Reduce contention with token-expert sorting and chunked reductions

- Launch granularity
  - Small `T` decode workloads and very large `T` prefill-like workloads both exist
  - Keep separate code paths or autotuned launch configs for small vs large `T`

## 8) Correctness validation plan (must do before tuning)

1. Compare routing outputs against reference only (`topk_idx`, `topk_w`)
2. Compare per-expert intermediate outputs after GEMM1 and after SwiGLU
3. Compare final output with tolerances from benchmark
4. Run with edge `seq_len` from workload file: tiny (1, 7), medium, huge (10k+)

Recommended checks:

- Exact match for selected expert ids
- Close match for route weights and output (`bf16` tolerance)
- No NaN/Inf in sigmoid/dequant/GEMM path

## 9) FlashInfer-Bench execution loop

1. Set solution definition to exact MoE definition name in `config.toml`
2. Implement kernel + binding entrypoint
3. `python scripts/pack_solution.py`
4. `python scripts/run_local.py`
5. Fix correctness first, then optimize
6. Repeat with profiling (nsight/ncu) once correct

## 10) Suggested milestone plan

- Milestone 1: routing kernel parity
- Milestone 2: dispatch + one-expert compute parity
- Milestone 3: full local experts parity
- Milestone 4: pass all workloads in local benchmark
- Milestone 5: optimize for latency and throughput

## 11) Common pitfalls

- Using biased scores for normalization (wrong): normalize using `s`, not `s_with_bias`
- Wrong scale layout indexing (`[H/128, T]` for hidden scale is transposed)
- Ignoring `local_expert_offset` in global expert id mapping
- Full dequant materialization exploding memory bandwidth
- FP16 accumulation causing drift on large `seq_len`

---

If you follow this staged plan, you can first land a correct solution quickly, then move to aggressive fusion and architecture-specific tuning.
