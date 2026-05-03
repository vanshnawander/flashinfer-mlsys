## submission-9

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- Results log: /home/vansh-nawander/Videos/flashinfer-mlsys/logs/benchmark_results_20260328_101437.log

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.233 ms | 3.55x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.697 ms | 6.41x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 9.567 ms | 1.45x | 2.56e+02 | 7.14e-03 |
| 8f1ff9f1... | 80 | PASSED | 13.221 ms | 1.21x | 5.12e+02 | 1.55e-02 |
| 1a4c6ba1... | 901 | PASSED | 13.534 ms | 1.54x | 4.10e+03 | 1.74e+03 |
| a7c2bcfd... | 16 | PASSED | 5.907 ms | 2.12x | 5.12e+02 | 7.09e-03 |
| 2e69caee... | 15 | PASSED | 2.703 ms | 4.20x | 1.56e-02 | 4.20e-03 |
| 8cba5890... | 14 | PASSED | 5.314 ms | 2.30x | 2.56e+02 | 7.46e-03 |
| 5e8dc11c... | 14107 | PASSED | 19.486 ms | 2.33x | 8.19e+03 | 3.55e+09 |
| 58a34f27... | 11948 | PASSED | 15.807 ms | 2.26x | 8.19e+03 | 1.10e+08 |
| 5eadab1e... | 62 | PASSED | 8.426 ms | 1.63x | 1.02e+03 | 1.54e-02 |
| eedc63b2... | 59 | PASSED | 8.546 ms | 1.59x | 1.02e+03 | 7.14e-03 |
| e626d3e6... | 58 | PASSED | 12.076 ms | 1.28x | 1.02e+03 | 1.32e-02 |
| 74d7ff04... | 57 | PASSED | 11.494 ms | 1.30x | 5.12e+02 | 1.56e-02 |
| 4822167c... | 56 | PASSED | 11.482 ms | 1.30x | 5.12e+02 | 7.69e-03 |
| 81955b1e... | 55 | PASSED | 10.849 ms | 1.34x | 1.02e+03 | 7.81e-03 |
| 76010cb4... | 54 | PASSED | 10.312 ms | 1.38x | 2.05e+03 | 1.19e-02 |
| fc378037... | 53 | PASSED | 10.924 ms | 1.34x | 1.02e+03 | 6.45e-03 |
| f7d6ac7c... | 52 | PASSED | 7.583 ms | 1.74x | 5.12e+02 | 4.44e-02 |

**Summary: 19/19 PASSED**