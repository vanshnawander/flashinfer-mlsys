## submission-1

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 4.136 ms | 2.86x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 2.570 ms | 4.35x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 10.449 ms | 1.36x | 2.05e+03 | 6.49e-02 |
| 8f1ff9f1... | 80 | PASSED | 14.188 ms | 1.13x | 5.12e+02 | 1.56e-02 |
| 1a4c6ba1... | 901 | PASSED | 20.561 ms | 1.04x | 2.05e+03 | 6.67e-01 |
| a7c2bcfd... | 16 | PASSED | 6.771 ms | 1.91x | 5.12e+02 | 6.76e-03 |
| 2e69caee... | 15 | PASSED | 3.607 ms | 3.24x | 6.40e+01 | 6.06e-03 |
| 8cba5890... | 14 | PASSED | 6.239 ms | 2.04x | 1.02e+03 | 7.35e-03 |
| 5e8dc11c... | 14107 | PASSED | 44.072 ms | 1.03x | 2.05e+03 | 1.00e+00 |
| 58a34f27... | 11948 | PASSED | 34.816 ms | 1.03x | 2.05e+03 | 7.81e+05 |
| 5eadab1e... | 62 | PASSED | 9.174 ms | 1.52x | 1.02e+03 | 3.23e-02 |
| eedc63b2... | 59 | PASSED | 9.301 ms | 1.48x | 2.56e+02 | 5.92e-03 |
| e626d3e6... | 58 | PASSED | 12.984 ms | 1.19x | 1.02e+03 | 1.50e-02 |
| 74d7ff04... | 57 | PASSED | 12.183 ms | 1.24x | 5.12e+02 | 8.20e-03 |
| 4822167c... | 56 | PASSED | 12.469 ms | 1.22x | 5.12e+02 | 7.75e-03 |
| 81955b1e... | 55 | PASSED | 11.620 ms | 1.27x | 1.02e+03 | 1.43e-01 |
| 76010cb4... | 54 | PASSED | 11.039 ms | 1.31x | 5.12e+02 | 7.69e-03 |
| fc378037... | 53 | PASSED | 11.603 ms | 1.27x | 5.12e+02 | 1.18e-02 |
| f7d6ac7c... | 52 | PASSED | 8.437 ms | 1.60x | 2.56e+02 | 7.52e-03 |

---

Template for future submissions:

- Use a section header `## submission-N` per submission.
- Include `Platform:` and `Benchmark:` metadata lines.
- Provide a table with columns: `Workload`, `Sequence Length`, `Status`, `Time`, `Speedup`, `abs_err`, `rel_err`.

Add additional submissions below following the same structure.

## submission-2

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 4.172 ms | 2.80x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 2.583 ms | 4.29x | 3.20e+01 | 6.37e-03 |
| 6230e838... | 32 | PASSED | 10.316 ms | 1.35x | 2.56e+02 | 7.35e-03 |
| 8f1ff9f1... | 80 | PASSED | 13.996 ms | 1.13x | 5.12e+02 | 3.45e-02 |
| 1a4c6ba1... | 901 | PASSED | 9.294 ms | 2.27x | 4.10e+03 | 1.54e+04 |
| a7c2bcfd... | 16 | PASSED | 6.727 ms | 1.88x | 5.12e+02 | 7.75e-03 |
| 2e69caee... | 15 | PASSED | 3.565 ms | 3.23x | 6.25e-02 | 5.62e-03 |
| 8cba5890... | 14 | PASSED | 6.160 ms | 2.01x | 6.40e+01 | 1.41e-02 |
| 5e8dc11c... | 14107 | PASSED | 27.104 ms | 1.66x | 8.19e+03 | 1.04e+08 |
| 58a34f27... | 11948 | PASSED | 20.281 ms | 1.77x | 4.10e+03 | 1.28e+09 |
| f7d6ac7c... | 52 | PASSED | 8.455 ms | 1.57x | 1.28e+02 | 5.81e-03 |


## submission-3

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.172 ms | 3.73x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.716 ms | 6.50x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 9.082 ms | 1.58x | 1.28e+02 | 6.90e-03 |
| 8f1ff9f1... | 80 | PASSED | 12.601 ms | 1.25x | 1.02e+03 | 3.17e-02 |
| 1a4c6ba1... | 901 | PASSED | 10.062 ms | 2.09x | 8.19e+03 | 1.08e+04 |
| a7c2bcfd... | 16 | PASSED | 5.641 ms | 2.25x | 1.00e+00 | 5.05e-03 |
| 2e69caee... | 15 | PASSED | 2.626 ms | 4.38x | 1.00e+00 | 7.14e-03 |
| 8cba5890... | 14 | PASSED | 5.134 ms | 2.43x | 2.56e+02 | 7.69e-03 |
| 5e8dc11c... | 14107 | PASSED | 24.363 ms | 1.85x | 8.19e+03 | 3.52e+09 |
| 58a34f27... | 11948 | PASSED | 17.646 ms | 2.03x | 8.19e+03 | 4.84e+07 |
| 5eadab1e... | 62 | PASSED | 8.021 ms | 1.72x | 2.56e+02 | 1.25e-02 |
| eedc63b2... | 59 | PASSED | 8.243 ms | 1.72x | 1.02e+03 | 6.71e-03 |
| e626d3e6... | 58 | PASSED | 11.574 ms | 1.32x | 1.02e+03 | 2.00e-02 |
| 74d7ff04... | 57 | PASSED | 10.908 ms | 1.42x | 5.12e+02 | 2.44e-02 |
| 4822167c... | 56 | PASSED | 11.111 ms | 1.41x | 1.02e+03 | 7.63e-03 |
| 81955b1e... | 55 | PASSED | 10.334 ms | 1.45x | 2.05e+03 | 7.69e-03 |
| 76010cb4... | 54 | PASSED | 9.715 ms | 1.48x | 5.12e+02 | 7.69e-03 |
| fc378037... | 53 | PASSED | 10.265 ms | 1.42x | 5.12e+02 | 1.33e-02 |
| f7d6ac7c... | 52 | PASSED | 7.200 ms | 1.85x | 1.02e+03 | 7.25e-03 |


## submission-5

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.254 ms | 3.63x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.753 ms | 6.40x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 9.181 ms | 1.57x | 1.60e+01 | 7.30e-03 |
| 8f1ff9f1... | 80 | PASSED | 12.692 ms | 1.27x | 5.12e+02 | 1.08e-02 |
| 1a4c6ba1... | 901 | PASSED | 18.448 ms | 1.15x | 1.02e+03 | 1.07e-01 |
| a7c2bcfd... | 16 | PASSED | 5.687 ms | 2.28x | 5.12e+02 | 6.90e-03 |
| 2e69caee... | 15 | PASSED | 2.654 ms | 4.40x | 6.25e-02 | 5.81e-03 |
| 8cba5890... | 14 | PASSED | 5.169 ms | 2.47x | 2.56e+02 | 5.59e-03 |
| 5e8dc11c... | 14107 | PASSED | 41.532 ms | 1.09x | 2.05e+03 | 7.81e+05 |
| 58a34f27... | 11948 | PASSED | 32.438 ms | 1.11x | 2.05e+03 | 2.34e+06 |
| 5eadab1e... | 62 | PASSED | 8.069 ms | 1.75x | 2.05e+03 | 1.41e-02 |
| eedc63b2... | 59 | PASSED | 8.166 ms | 1.72x | 1.02e+03 | 7.35e-03 |
| e626d3e6... | 58 | PASSED | 11.618 ms | 1.34x | 1.02e+03 | 7.46e-03 |
| 74d7ff04... | 57 | PASSED | 10.874 ms | 1.40x | 5.12e+02 | 7.52e-03 |
| 4822167c... | 56 | PASSED | 11.136 ms | 1.38x | 1.02e+03 | 1.00e+00 |
| 81955b1e... | 55 | PASSED | 10.310 ms | 1.45x | 1.02e+03 | 7.81e-03 |
| 76010cb4... | 54 | PASSED | 9.785 ms | 1.50x | 5.12e+02 | 7.75e-03 |
| fc378037... | 53 | PASSED | 10.357 ms | 1.45x | 1.28e+02 | 1.85e-02 |
| f7d6ac7c... | 52 | PASSED | 7.263 ms | 1.88x | 2.56e+02 | 7.25e-03 |


submission 6:
## submission-6

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.334 ms | 3.54x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.820 ms | 6.14x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 9.328 ms | 1.54x | 5.12e+02 | 7.75e-03 |
| 8f1ff9f1... | 80 | PASSED | 12.670 ms | 1.24x | 4.10e+03 | 1.57e+02 |
| 1a4c6ba1... | 901 | PASSED | 8.698 ms | 2.41x | 4.10e+03 | 2.72e+03 |
| a7c2bcfd... | 16 | PASSED | 5.741 ms | 2.19x | 5.12e+02 | 6.54e-03 |
| 2e69caee... | 15 | PASSED | 2.827 ms | 4.08x | 1.02e+03 | 6.45e-03 |
| 8cba5890... | 14 | PASSED | 5.299 ms | 2.39x | 1.95e-03 | 4.65e-02 |
| 5e8dc11c... | 14107 | PASSED | 19.281 ms | 2.35x | 8.19e+03 | 2.14e+04 |
| 58a34f27... | 11948 | PASSED | 14.595 ms | 2.45x | 8.19e+03 | 1.25e+09 |
| 5eadab1e... | 62 | PASSED | 7.700 ms | 1.76x | 4.10e+03 | 8.50e+02 |
| eedc63b2... | 59 | PASSED | 8.184 ms | 1.63x | 1.02e+03 | 7.14e-03 |
| e626d3e6... | 58 | PASSED | 11.495 ms | 1.32x | 4.10e+03 | 6.42e+02 |
| 74d7ff04... | 57 | PASSED | 10.742 ms | 1.36x | 4.10e+03 | 1.73e+02 |
| 4822167c... | 56 | PASSED | 11.069 ms | 1.33x | 2.05e+03 | 6.13e+02 |
| 81955b1e... | 55 | PASSED | 10.438 ms | 1.37x | 2.05e+03 | 6.21e-03 |
| 76010cb4... | 54 | PASSED | 9.877 ms | 1.42x | 2.05e+03 | 1.39e-02 |
| fc378037... | 53 | PASSED | 10.423 ms | 1.38x | 5.12e+02 | 5.00e-01 |
| f7d6ac7c... | 52 | PASSED | 7.313 ms | 1.78x | 8.00e+00 | 4.44e-03 |


Submission - 8

Had numerical issues 1 case passed code for that is in working-numerical-issue.py 





## submission-9

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 2.840 ms | 3.81x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.390 ms | 7.43x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 8.703 ms | 1.48x | 1.28e+02 | 6.13e-03 |
| 8f1ff9f1... | 80 | PASSED | 12.076 ms | 1.23x | 1.02e+03 | 7.81e-03 |
| 1a4c6ba1... | 901 | PASSED | 12.450 ms | 1.59x | 4.10e+03 | 2.89e+04 |
| a7c2bcfd... | 16 | PASSED | 5.276 ms | 2.21x | 5.12e+02 | 7.81e-03 |
| 2e69caee... | 15 | PASSED | 2.288 ms | 4.69x | 5.12e+02 | 6.71e-03 |
| 8cba5890... | 14 | PASSED | 4.750 ms | 2.40x | 0.00e+00 | 0.00e+00 |
| 5e8dc11c... | 14107 | PASSED | 18.540 ms | 2.35x | 4.10e+03 | 8.05e+09 |
| 58a34f27... | 11948 | PASSED | 14.886 ms | 2.32x | 4.10e+03 | 5.90e+09 |
| 5eadab1e... | 62 | PASSED | 7.654 ms | 1.67x | 5.12e+02 | 2.04e-02 |
| eedc63b2... | 59 | PASSED | 7.751 ms | 1.62x | 1.02e+03 | 7.75e-03 |
| e626d3e6... | 58 | PASSED | 11.010 ms | 1.29x | 1.02e+03 | 2.06e-02 |
| 74d7ff04... | 57 | PASSED | 10.443 ms | 1.31x | 1.02e+03 | 2.82e-02 |
| 4822167c... | 56 | PASSED | 10.544 ms | 1.30x | 5.12e+02 | 1.25e-02 |
| 81955b1e... | 55 | PASSED | 9.866 ms | 1.36x | 1.02e+03 | 7.69e-03 |
| 76010cb4... | 54 | PASSED | 9.384 ms | 1.40x | 2.05e+03 | 1.20e-02 |
| fc378037... | 53 | PASSED | 9.891 ms | 1.36x | 5.12e+02 | 6.67e-03 |
| f7d6ac7c... | 52 | PASSED | 6.861 ms | 1.78x | 5.12e+02 | 5.65e-03 |



## submission-10 - fixed

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 2.866 ms | 3.80x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.381 ms | 7.53x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 8.716 ms | 1.48x | 1.02e+03 | 7.14e-03 |
| 8f1ff9f1... | 80 | PASSED | 12.034 ms | 1.23x | 1.02e+03 | 1.34e-02 |
| 1a4c6ba1... | 901 | RUNTIME_ERROR | - | - | - | - |
| a7c2bcfd... | 16 | PASSED | 5.335 ms | 2.18x | 5.12e+02 | 7.25e-03 |
| 2e69caee... | 15 | PASSED | 2.262 ms | 4.75x | 7.81e-03 | 5.62e-03 |
| 8cba5890... | 14 | PASSED | 4.801 ms | 2.37x | 1.28e+02 | 6.10e-03 |
| 5e8dc11c... | 14107 | RUNTIME_ERROR | - | - | - | - |
| 58a34f27... | 11948 | RUNTIME_ERROR | - | - | - | - |
| 5eadab1e... | 62 | PASSED | 7.613 ms | 1.68x | 5.12e+02 | 1.06e-02 |
| eedc63b2... | 59 | PASSED | 7.780 ms | 1.62x | 1.02e+03 | 6.54e-03 |
| e626d3e6... | 58 | PASSED | 11.060 ms | 1.29x | 1.02e+03 | 7.30e-03 |
| 74d7ff04... | 57 | PASSED | 10.386 ms | 1.32x | 1.02e+03 | 3.28e-02 |
| 4822167c... | 56 | PASSED | 10.537 ms | 1.31x | 1.02e+03 | 3.70e-02 |
| 81955b1e... | 55 | PASSED | 9.837 ms | 1.37x | 1.02e+03 | 7.69e-03 |
| 76010cb4... | 54 | PASSED | 9.313 ms | 1.42x | 5.12e+02 | 7.14e-03 |
| fc378037... | 53 | PASSED | 9.872 ms | 1.37x | 2.56e+02 | 2.41e-02 |
| f7d6ac7c... | 52 | PASSED | 6.816 ms | 1.80x | 2.56e+02 | 5.92e-03 |



## submission-10

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.206 ms | 3.62x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.758 ms | 6.21x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 9.469 ms | 1.46x | 4.00e+00 | 4.50e-03 |
| 8f1ff9f1... | 80 | PASSED | 12.800 ms | 1.22x | 1.02e+03 | 2.14e-02 |
| 1a4c6ba1... | 901 | PASSED | 12.594 ms | 1.66x | 8.19e+03 | 7.97e+03 |
| a7c2bcfd... | 16 | PASSED | 5.803 ms | 2.15x | 1.02e+03 | 7.41e-03 |
| 2e69caee... | 15 | PASSED | 2.655 ms | 4.27x | 8.00e+00 | 5.35e-03 |
| 8cba5890... | 14 | PASSED | 5.268 ms | 2.31x | 1.60e+01 | 5.21e-03 |
| 5e8dc11c... | 14107 | PASSED | 17.324 ms | 2.58x | 8.19e+03 | 1.40e+09 |
| 58a34f27... | 11948 | PASSED | 13.803 ms | 2.57x | 8.19e+03 | 1.56e+09 |
| 5eadab1e... | 62 | PASSED | 8.322 ms | 1.63x | 1.02e+03 | 1.47e-02 |
| eedc63b2... | 59 | PASSED | 8.426 ms | 1.58x | 5.12e+02 | 7.69e-03 |
| e626d3e6... | 58 | PASSED | 11.683 ms | 1.30x | 1.02e+03 | 1.89e-02 |
| 74d7ff04... | 57 | PASSED | 11.232 ms | 1.32x | 1.02e+03 | 1.37e-02 |
| 4822167c... | 56 | PASSED | 11.233 ms | 1.31x | 5.12e+02 | 2.94e-02 |
| 81955b1e... | 55 | PASSED | 10.690 ms | 1.34x | 5.12e+02 | 8.33e-02 |
| 76010cb4... | 54 | PASSED | 10.105 ms | 1.40x | 1.02e+03 | 2.36e-02 |
| fc378037... | 53 | PASSED | 10.746 ms | 1.34x | 5.12e+02 | 1.25e-01 |
| f7d6ac7c... | 52 | PASSED | 7.501 ms | 1.74x | 2.56e+02 | 5.92e-03 |


## submission-12

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.267 ms | 3.50x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.691 ms | 6.43x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 9.541 ms | 1.43x | 1.02e+03 | 8.93e-03 |
| 8f1ff9f1... | 80 | PASSED | 13.197 ms | 1.19x | 1.02e+03 | 7.75e-03 |
| 1a4c6ba1... | 901 | RUNTIME_ERROR | - | - | - | - |
| a7c2bcfd... | 16 | PASSED | 5.859 ms | 2.13x | 5.12e+02 | 7.19e-03 |
| 2e69caee... | 15 | PASSED | 2.651 ms | 4.29x | 1.60e+01 | 4.74e-03 |
| 8cba5890... | 14 | PASSED | 5.303 ms | 2.29x | 1.02e+03 | 5.88e-03 |
| 5e8dc11c... | 14107 | RUNTIME_ERROR | - | - | - | - |
| 58a34f27... | 11948 | RUNTIME_ERROR | - | - | - | - |
| 5eadab1e... | 62 | PASSED | 8.347 ms | 1.63x | 5.12e+02 | 2.74e-02 |
| eedc63b2... | 59 | PASSED | 8.472 ms | 1.58x | 5.12e+02 | 7.09e-03 |
| e626d3e6... | 58 | PASSED | 12.101 ms | 1.25x | 5.12e+02 | 1.49e-02 |
| 74d7ff04... | 57 | PASSED | 11.318 ms | 1.30x | 1.02e+03 | 1.48e-02 |
| 4822167c... | 56 | PASSED | 11.627 ms | 1.27x | 2.56e+02 | 3.57e-02 |
| 81955b1e... | 55 | PASSED | 10.803 ms | 1.33x | 2.56e+02 | 1.96e-02 |
| 76010cb4... | 54 | PASSED | 10.211 ms | 1.38x | 2.05e+03 | 7.25e-03 |
| fc378037... | 53 | PASSED | 10.779 ms | 1.34x | 1.02e+03 | 3.33e-02 |
| f7d6ac7c... | 52 | PASSED | 7.540 ms | 1.73x | 5.12e+02 | 6.06e-03 |

Results successfully saved to 
/home/vanshnawander/accelerated-hpc/flashinfer-mlsys/logs/benchmark_results_20260323_123435.log

## submission-13

- Platform: Modal B200
- Benchmark: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

| Workload | Sequence Length | Status | Time | Speedup | abs_err | rel_err |
|---|---:|:---:|---:|---:|---:|---:|
| b8f4f012... | 7 | PASSED | 3.263 ms | 3.53x | 0.00e+00 | 0.00e+00 |
| e05c6c03... | 1 | PASSED | 1.722 ms | 6.34x | 0.00e+00 | 0.00e+00 |
| 6230e838... | 32 | PASSED | 9.636 ms | 1.44x | 1.02e+03 | 7.19e-03 |
| 8f1ff9f1... | 80 | PASSED | 13.603 ms | 1.17x | 1.02e+03 | 7.09e-03 |
| 1a4c6ba1... | 901 | PASSED | 13.544 ms | 1.54x | 4.10e+03 | 2.11e+03 |
| a7c2bcfd... | 16 | PASSED | 5.943 ms | 2.11x | 3.20e+01 | 5.99e-03 |
| 2e69caee... | 15 | PASSED | 2.736 ms | 4.17x | 0.00e+00 | 0.00e+00 |
| 8cba5890... | 14 | PASSED | 5.359 ms | 2.29x | 1.60e+01 | 6.54e-03 |
| 5e8dc11c... | 14107 | PASSED | 18.997 ms | 2.39x | 8.19e+03 | 3.35e+09 |
| 58a34f27... | 11948 | PASSED | 15.514 ms | 2.32x | 8.19e+03 | 5.85e+09 |
| 5eadab1e... | 62 | PASSED | 8.614 ms | 1.61x | 5.12e+02 | 7.81e-03 |
| eedc63b2... | 59 | PASSED | 8.746 ms | 1.58x | 2.56e+02 | 6.41e-03 |
| e626d3e6... | 58 | PASSED | 12.517 ms | 1.24x | 1.02e+03 | 4.76e-02 |
| 74d7ff04... | 57 | PASSED | 11.546 ms | 1.29x | 5.12e+02 | 2.68e-02 |
| 4822167c... | 56 | PASSED | 11.834 ms | 1.26x | 5.12e+02 | 1.85e-02 |
| 81955b1e... | 55 | PASSED | 10.971 ms | 1.33x | 1.02e+03 | 2.22e-02 |
| 76010cb4... | 54 | PASSED | 10.409 ms | 1.37x | 5.12e+02 | 2.70e-02 |
| fc378037... | 53 | PASSED | 11.002 ms | 1.33x | 5.12e+02 | 1.02e-02 |
| f7d6ac7c... | 52 | PASSED | 7.644 ms | 1.73x | 1.28e+02 | 8.00e-03 |

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





