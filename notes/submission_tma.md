## submission-tma

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- Results log: /home/vanshnawander/accelerated-hpc/flashinfer-mlsys/logs/benchmark_results_20260403_185750.log

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.654 ms | 3.16x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.930 ms | 5.86x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 10.045 ms | 1.43x | 5.12e+02 | 8.70e-03 |
| 8f1ff9f1... | 80 | PASSED | 14.205 ms | 1.14x | 1.02e+03 | 7.69e-03 |
| 1a4c6ba1... | 901 | RUNTIME_ERROR | - | - | - | - |
| a7c2bcfd... | 16 | PASSED | 6.217 ms | 2.04x | 1.02e+03 | 6.94e-03 |
| 2e69caee... | 15 | PASSED | 2.909 ms | 3.94x | 6.40e+01 | 6.49e-03 |
| 8cba5890... | 14 | PASSED | 5.627 ms | 2.20x | 0.00e+00 | 0.00e+00 |
| 5e8dc11c... | 14107 | RUNTIME_ERROR | - | - | - | - |
| 58a34f27... | 11948 | RUNTIME_ERROR | - | - | - | - |
| 5eadab1e... | 62 | PASSED | 8.911 ms | 1.55x | 1.02e+03 | 7.75e-03 |
| eedc63b2... | 59 | PASSED | 9.032 ms | 1.53x | 5.12e+02 | 9.43e-03 |
| e626d3e6... | 58 | PASSED | 12.998 ms | 1.19x | 5.12e+02 | 7.46e-03 |
| 74d7ff04... | 57 | PASSED | 12.066 ms | 1.25x | 5.12e+02 | 7.30e-03 |
| 4822167c... | 56 | PASSED | 12.316 ms | 1.23x | 1.02e+03 | 8.10e-03 |
| 81955b1e... | 55 | PASSED | 11.356 ms | 1.29x | 1.02e+03 | 7.81e-03 |
| 76010cb4... | 54 | PASSED | 10.825 ms | 1.33x | 1.02e+03 | 7.25e-03 |
| fc378037... | 53 | PASSED | 11.505 ms | 1.28x | 2.56e+02 | 1.33e-02 |
| f7d6ac7c... | 52 | PASSED | 8.038 ms | 1.66x | 1.02e+03 | 2.04e-02 |

**Summary: 16/19 PASSED**