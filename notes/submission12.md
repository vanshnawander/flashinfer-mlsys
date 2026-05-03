
## submission-12

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 2.898 ms | 3.77x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.444 ms | 7.18x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 8.887 ms | 1.47x | 5.12e+02 | 6.25e-03 |
| 8f1ff9f1... | 80 | PASSED | 12.371 ms | 1.24x | 5.12e+02 | 6.67e-03 |
| 1a4c6ba1... | 901 | RUNTIME_ERROR | - | - | - | - |
| a7c2bcfd... | 16 | PASSED | 5.405 ms | 2.19x | 4.00e+00 | 9.90e-03 |
| 2e69caee... | 15 | PASSED | 2.381 ms | 4.56x | 2.56e+02 | 1.45e-02 |
| 8cba5890... | 14 | PASSED | 4.845 ms | 2.37x | 8.00e+00 | 7.19e-03 |
| 5e8dc11c... | 14107 | RUNTIME_ERROR | - | - | - | - |
| 58a34f27... | 11948 | RUNTIME_ERROR | - | - | - | - |
| 5eadab1e... | 62 | PASSED | 7.789 ms | 1.67x | 5.12e+02 | 1.54e-02 |
| eedc63b2... | 59 | PASSED | 7.929 ms | 1.62x | 1.02e+03 | 7.25e-03 |
| e626d3e6... | 58 | PASSED | 11.274 ms | 1.30x | 1.02e+03 | 7.63e-03 |
| 74d7ff04... | 57 | PASSED | 10.692 ms | 1.32x | 1.28e+02 | 1.72e-02 |
| 4822167c... | 56 | PASSED | 10.651 ms | 1.33x | 1.02e+03 | 6.85e-03 |
| 81955b1e... | 55 | PASSED | 10.098 ms | 1.36x | 5.12e+02 | 8.66e-03 |
| 76010cb4... | 54 | PASSED | 9.584 ms | 1.41x | 1.28e+02 | 7.69e-03 |
| fc378037... | 53 | PASSED | 10.142 ms | 1.36x | 1.02e+03 | 7.69e-03 |
| f7d6ac7c... | 52 | PASSED | 6.987 ms | 1.78x | 5.12e+02 | 7.41e-03 |