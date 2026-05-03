## submission-13-cuda

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- Results log: /home/vanshnawander/accelerated-hpc/flashinfer-mlsys/logs/benchmark_results_20260326_182555.log

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.094 ms | 3.80x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.579 ms | 6.95x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 9.259 ms | 1.53x | 1.25e-01 | 6.67e-03 |
| 8f1ff9f1... | 80 | PASSED | 13.101 ms | 1.22x | 2.05e+03 | 7.75e-03 |
| 1a4c6ba1... | 901 | PASSED | 18.753 ms | 1.10x | 1.02e+03 | 6.67e-02 |
| a7c2bcfd... | 16 | PASSED | 5.681 ms | 2.21x | 5.12e+02 | 1.18e-02 |
| 2e69caee... | 15 | PASSED | 2.556 ms | 4.44x | 8.00e+00 | 5.88e-03 |
| 8cba5890... | 14 | PASSED | 5.095 ms | 2.40x | 3.20e+01 | 7.19e-03 |
| 5e8dc11c... | 14107 | PASSED | 42.806 ms | 1.05x | 2.05e+03 | 7.81e+05 |
| 58a34f27... | 11948 | PASSED | 33.291 ms | 1.07x | 2.05e+03 | 1.00e+00 |
| 5eadab1e... | 62 | PASSED | 8.107 ms | 1.69x | 1.02e+03 | 7.04e-03 |
| eedc63b2... | 59 | PASSED | 8.232 ms | 1.65x | 5.12e+02 | 3.38e-02 |
| e626d3e6... | 58 | PASSED | 11.926 ms | 1.29x | 1.02e+03 | 5.13e-02 |
| 74d7ff04... | 57 | PASSED | 11.082 ms | 1.34x | 5.12e+02 | 6.99e-03 |
| 4822167c... | 56 | PASSED | 11.337 ms | 1.32x | 1.02e+03 | 1.23e-02 |
| 81955b1e... | 55 | PASSED | 10.472 ms | 1.40x | 1.02e+03 | 1.45e-02 |
| 76010cb4... | 54 | PASSED | 9.942 ms | 1.44x | 6.40e+01 | 4.74e-03 |
| fc378037... | 53 | PASSED | 10.532 ms | 1.39x | 2.05e+03 | 1.08e-02 |
| f7d6ac7c... | 52 | PASSED | 7.292 ms | 1.81x | 5.12e+02 | 5.13e-03 |

## submission-16-cuda

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- Results log: /home/vanshnawander/accelerated-hpc/flashinfer-mlsys/logs/benchmark_results_20260326_192205.log

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.059 ms | 3.87x | 1.00e+00 | 5.92e-03 |
| e05c6c03... | 1 | PASSED | 1.577 ms | 7.04x | 3.91e-03 | 7.14e-02 |
| 6230e838... | 32 | PASSED | 9.181 ms | 1.55x | 2.56e+02 | 2.48e-02 |
| 8f1ff9f1... | 80 | PASSED | 12.759 ms | 1.27x | 1.02e+03 | 8.71e-02 |
| 1a4c6ba1... | 901 | PASSED | 18.516 ms | 1.14x | 2.05e+03 | 5.00e-01 |
| a7c2bcfd... | 16 | PASSED | 5.622 ms | 2.28x | 5.12e+02 | 6.90e-03 |
| 2e69caee... | 15 | PASSED | 2.535 ms | 4.54x | 3.20e+01 | 6.85e-03 |
| 8cba5890... | 14 | PASSED | 5.050 ms | 2.48x | 2.56e+02 | 6.10e-03 |
| 5e8dc11c... | 14107 | PASSED | 41.967 ms | 1.09x | 2.05e+03 | 1.09e+00 |
| 58a34f27... | 11948 | PASSED | 32.317 ms | 1.11x | 4.10e+03 | 3.91e+05 |
| 5eadab1e... | 62 | PASSED | 8.037 ms | 1.74x | 5.12e+02 | 9.84e-02 |
| eedc63b2... | 59 | PASSED | 8.140 ms | 1.72x | 1.02e+03 | 7.35e-03 |
| e626d3e6... | 58 | PASSED | 11.596 ms | 1.35x | 5.12e+02 | 2.06e-02 |
| 74d7ff04... | 57 | PASSED | 10.964 ms | 1.38x | 1.02e+03 | 1.33e-01 |
| 4822167c... | 56 | PASSED | 10.928 ms | 1.39x | 5.12e+02 | 4.05e-02 |
| 81955b1e... | 55 | PASSED | 10.360 ms | 1.42x | 5.12e+02 | 7.87e-03 |
| 76010cb4... | 54 | PASSED | 9.842 ms | 1.47x | 5.12e+02 | 7.04e-03 |
| fc378037... | 53 | PASSED | 10.447 ms | 1.42x | 1.28e+02 | 7.14e-03 |
| f7d6ac7c... | 52 | PASSED | 7.206 ms | 1.87x | 1.60e+01 | 6.25e-03 |