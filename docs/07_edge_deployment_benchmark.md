# 07 — Kısıtlı Edge / Docker Dağıtım Benchmark Raporu

- **Konteyner Ortamı:** `Linux-6.18.40.1-microsoft-standard-WSL2-x86_64-with-glibc2.41` (`x86_64`)
- **Derleme Stratejisi:** Multi-stage build (`gcc (Debian 14.2.0-19) 14.2.0`, `-O3`); çalışma imajında derleyici bulunmaz.
- **Test Örneklemi:** `Merged09.csv` kilitli setinden 10.000 gerçek akış (2.000 Benign + 8.000 Saldırı)
- **Bellek Ölçümü:** Linux çekirdeği `/proc/self/status` → `VmHWM` (Peak Resident Set Size)

## Tüm Kısıtlı Profillerde ONNX Runtime vs. Saf C99 Kıyaslaması

| profile | cpu_limit | ram_limit_mb | model | onnx_size_kb | linux_bin_kb | onnx_peak_ram_mb | c_peak_ram_mb | ram_ratio_onnx_vs_c_x | onnx_p50_us | c_p50_us | c_p99_us | speedup_c_vs_onnx_x | c_pps | c_vs_onnx_agree_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25cpu_128mb | 0.25 | 128.0 | top15_l0_baseline | 673.58 | 436.8 | 64.492 | 2.574 | 25.1 | 39.05 | 0.9 | 26.8 | 43.4 | 91325.6 | 100.0 |
| 0.25cpu_128mb | 0.25 | 128.0 | top15_l1_balanced | 174.97 | 157.82 | 58.023 | 2.266 | 25.6 | 31.15 | 0.3 | 3.7 | 103.8 | 505394.6 | 100.0 |
| 0.25cpu_128mb | 0.25 | 128.0 | top10_l2_light | 52.63 | 72.6 | 56.723 | 2.141 | 26.5 | 27.7 | 0.2 | 1.1 | 138.5 | 7312507.8 | 100.0 |
| 0.5cpu_256mb | 0.5 | 256.0 | top15_l0_baseline | 673.58 | 436.8 | 64.578 | 2.539 | 25.4 | 40.4 | 1.0 | 21.8 | 40.4 | 212637.5 | 100.0 |
| 0.5cpu_256mb | 0.5 | 256.0 | top15_l1_balanced | 174.97 | 157.82 | 58.457 | 2.25 | 26.0 | 32.25 | 0.3 | 2.8 | 107.5 | 1002115.5 | 100.0 |
| 0.5cpu_256mb | 0.5 | 256.0 | top10_l2_light | 52.63 | 72.6 | 56.766 | 2.23 | 25.5 | 36.6 | 0.1 | 1.1 | 366.0 | 1468006.3 | 100.0 |

- Ham CSV çıktıları: `results/edge_benchmark/edge_benchmark_*.csv`