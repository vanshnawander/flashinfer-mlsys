## Triton submission-9 vs CUDA submission-16

- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- Triton source: `notes/solution_benchmarks.md` (`submission-9`)
- CUDA source: `notes/solution_benchmarks_cuda.md` (`submission-16-cuda`)
- Sequence length mapping source: `logs/benchmark_results_20260326_151308.log`

| Workload | Sequence Length | Triton-9 Time | CUDA-16 Time | Delta (CUDA - Triton) | Faster |
|---|---:|---:|---:|---:|:---:|
| b8f4f012... | 7 | 2.840 ms | 3.059 ms | +0.219 ms | Triton |
| e05c6c03... | 1 | 1.390 ms | 1.577 ms | +0.187 ms | Triton |
| 6230e838... | 32 | 8.703 ms | 9.181 ms | +0.478 ms | Triton |
| 8f1ff9f1... | 80 | 12.076 ms | 12.759 ms | +0.683 ms | Triton |
| 1a4c6ba1... | 901 | 12.450 ms | 18.516 ms | +6.066 ms | Triton |
| a7c2bcfd... | 16 | 5.276 ms | 5.622 ms | +0.346 ms | Triton |
| 2e69caee... | 15 | 2.288 ms | 2.535 ms | +0.247 ms | Triton |
| 8cba5890... | 14 | 4.750 ms | 5.050 ms | +0.300 ms | Triton |
| 5e8dc11c... | 14107 | 18.540 ms | 41.967 ms | +23.427 ms | Triton |
| 58a34f27... | 11948 | 14.886 ms | 32.317 ms | +17.431 ms | Triton |
| 5eadab1e... | 62 | 7.654 ms | 8.037 ms | +0.383 ms | Triton |
| eedc63b2... | 59 | 7.751 ms | 8.140 ms | +0.389 ms | Triton |
| e626d3e6... | 58 | 11.010 ms | 11.596 ms | +0.586 ms | Triton |
| 74d7ff04... | 57 | 10.443 ms | 10.964 ms | +0.521 ms | Triton |
| 4822167c... | 56 | 10.544 ms | 10.928 ms | +0.384 ms | Triton |
| 81955b1e... | 55 | 9.866 ms | 10.360 ms | +0.494 ms | Triton |
| 76010cb4... | 54 | 9.384 ms | 9.842 ms | +0.458 ms | Triton |
| fc378037... | 53 | 9.891 ms | 10.447 ms | +0.556 ms | Triton |
| f7d6ac7c... | 52 | 6.861 ms | 7.206 ms | +0.345 ms | Triton |

## Key observations

- Triton submission-9 is faster on all 19/19 workloads.
- Total latency reduction across workloads is ~53.5 ms, average ~2.82 ms per workload.
- Largest gains are on very long sequences:
  - `14107`: Triton 18.540 ms vs CUDA 41.967 ms
  - `11948`: Triton 14.886 ms vs CUDA 32.317 ms
- Small/medium sequence lengths (1 to 80) show consistent but modest gains (roughly ~0.19 to ~0.68 ms each).

## Numerical behavior notes

- For long-sequence workloads (`14107`, `11948`), Triton-9 has much larger `rel_err` (`8.05e+09`, `5.90e+09`) than CUDA-16 (`1.09e+00`, `3.91e+05`).
- For sequence length `901`, Triton-9 `rel_err=2.89e+04` while CUDA-16 `rel_err=5.00e-01`.
- Conclusion: submission-9 (Triton) is clearly better on raw latency, but CUDA submission-16 is materially better on numerical stability for the hardest long-context cases.
