
## submission-9opt

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- Results log: /home/vansh-nawander/Videos/flashinfer-mlsys/logs/benchmark_results_20260328_102311.log

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 2.912 ms | 3.75x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.421 ms | 7.39x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 8.989 ms | 1.46x | 5.12e+02 | 8.13e-03 |
| 8f1ff9f1... | 80 | PASSED | 12.849 ms | 1.20x | 1.02e+03 | 1.67e-01 |
| 1a4c6ba1... | 901 | RUNTIME_ERROR | - | - | - | - |
| a7c2bcfd... | 16 | PASSED | 5.436 ms | 2.18x | 2.56e+02 | 6.76e-03 |
| 2e69caee... | 15 | PASSED | 2.365 ms | 4.62x | 1.28e+02 | 4.83e-03 |
| 8cba5890... | 14 | PASSED | 4.859 ms | 2.38x | 4.10e+03 | 2.70e-02 |
| 5e8dc11c... | 14107 | RUNTIME_ERROR | - | - | - | - |
| 58a34f27... | 11948 | RUNTIME_ERROR | - | - | - | - |
| 5eadab1e... | 62 | PASSED | 7.878 ms | 1.65x | 5.12e+02 | 8.33e-03 |
| eedc63b2... | 59 | PASSED | 7.998 ms | 1.61x | 5.12e+02 | 7.25e-03 |
| e626d3e6... | 58 | PASSED | 11.730 ms | 1.26x | 5.12e+02 | 5.36e-02 |
| 74d7ff04... | 57 | PASSED | 10.835 ms | 1.30x | 5.12e+02 | 3.11e-02 |
| 4822167c... | 56 | PASSED | 11.066 ms | 1.27x | 5.12e+02 | 7.58e-03 |
| 81955b1e... | 55 | PASSED | 10.215 ms | 1.34x | 5.12e+02 | 7.52e-03 |
| 76010cb4... | 54 | PASSED | 9.702 ms | 1.39x | 5.12e+02 | 8.93e-03 |
| fc378037... | 53 | PASSED | 10.299 ms | 1.34x | 1.02e+03 | 7.69e-03 |
| f7d6ac7c... | 52 | PASSED | 7.052 ms | 1.76x | 1.28e+02 | 5.38e-03 |

**Summary: 16/19 PASSED**