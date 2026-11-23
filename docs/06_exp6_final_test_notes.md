# Exp 6 — Kilitli Final Test Sonuçları (`Merged09.csv + Merged10.csv`)

- **Test Kümesi (Tek Seferlik, %100 Örneklemesiz):** `Merged09.csv, Merged10.csv` — Toplam **1,595,072** akış (**37,104** görülmemiş Benign, **1,557,968** Saldırı)
- **Akademik Bütünlük:** Bu aşamada hiçbir model yeniden eğitilmemiş, hiçbir özellik veya alarm eşiği değiştirilmemiştir. Tüm artefaktlar `models/final_edge/frozen_manifest.json` içinde SHA-256 ile mühürlenmiştir.

## 1. Ana Final Metrik Tablosu (Saf C99 Native)

| model             | role                   |   n_features | operating_point   |   locked_threshold_m07 |   accuracy |   f1_macro |   attack_precision |   attack_recall |   benign_precision |   benign_recall_specificity |   fpr_% |   fnr_% |   fp_per_10k_normal |      tp |   fp |    tn |    fn |   roc_auc |   pr_auc |   live99_precision |   live99_f1_attack |   live95_precision |   live95_f1_attack |
|:------------------|:-----------------------|-------------:|:------------------|-----------------------:|-----------:|-----------:|-------------------:|----------------:|-------------------:|----------------------------:|--------:|--------:|--------------------:|--------:|-----:|------:|------:|----------:|---------:|-------------------:|-------------------:|-------------------:|-------------------:|
| top15_l0_baseline | Referans (Budanmamis)  |           15 | Target_FPR_0.5%   |               0.738315 |    0.98672 |    0.88493 |            0.99983 |         0.98657 |            0.63782 |                     0.99288 |   0.712 |   1.343 |                71.2 | 1537049 |  264 | 36840 | 20919 |   0.99803 |  0.99995 |            0.58397 |            0.73411 |            0.87951 |            0.93004 |
| top15_l0_baseline | Referans (Budanmamis)  |           15 | Target_FPR_0.2%   |               0.84289  |    0.98522 |    0.87541 |            0.99994 |         0.98493 |            0.6118  |                     0.99752 |   0.248 |   1.507 |                24.8 | 1534483 |   92 | 37012 | 23485 |   0.99803 |  0.99995 |            0.80079 |            0.88388 |            0.95437 |            0.96953 |
| top15_l0_baseline | Referans (Budanmamis)  |           15 | Target_FPR_0.1%   |               0.88668  |    0.98442 |    0.8704  |            0.99996 |         0.98408 |            0.59904 |                     0.99854 |   0.146 |   1.592 |                14.6 | 1533169 |   54 | 37050 | 24799 |   0.99803 |  0.99995 |            0.8725  |            0.92552 |            0.97268 |            0.97852 |
| top15_l1_balanced | Ana Onerilen Model     |           15 | Target_FPR_0.5%   |               0.718811 |    0.98544 |    0.87669 |            0.99989 |         0.9852  |            0.61575 |                     0.99553 |   0.447 |   1.48  |                44.7 | 1534917 |  166 | 36938 | 23051 |   0.9977  |  0.99994 |            0.69031 |            0.81233 |            0.92059 |            0.9519  |
| top15_l1_balanced | Ana Onerilen Model     |           15 | Target_FPR_0.2%   |               0.857909 |    0.984   |    0.86776 |            0.99996 |         0.98366 |            0.59269 |                     0.99825 |   0.175 |   1.634 |                17.5 | 1532514 |   65 | 37039 | 25454 |   0.9977  |  0.99994 |            0.85039 |            0.91286 |            0.96728 |            0.97557 |
| top15_l1_balanced | Ana Onerilen Model     |           15 | Target_FPR_0.1%   |               0.915133 |    0.98311 |    0.86236 |            0.99997 |         0.98274 |            0.57946 |                     0.99889 |   0.111 |   1.726 |                11.1 | 1531070 |   41 | 37063 | 26898 |   0.9977  |  0.99994 |            0.90004 |            0.94035 |            0.97909 |            0.98104 |
| top10_l2_light    | Ultra-Hafif Alternatif |           10 | Target_FPR_0.5%   |               0.722452 |    0.98395 |    0.86722 |            0.9999  |         0.98367 |            0.59222 |                     0.99577 |   0.423 |   1.633 |                42.3 | 1532528 |  157 | 36947 | 25440 |   0.99706 |  0.99993 |            0.70176 |            0.81965 |            0.92447 |            0.95326 |
| top10_l2_light    | Ultra-Hafif Alternatif |           10 | Target_FPR_0.2%   |               0.891319 |    0.98037 |    0.84643 |            0.99996 |         0.97994 |            0.54245 |                     0.99854 |   0.146 |   2.006 |                14.6 | 1526717 |   54 | 37050 | 31251 |   0.99706 |  0.99993 |            0.87208 |            0.92368 |            0.97256 |            0.97629 |
| top10_l2_light    | Ultra-Hafif Alternatif |           10 | Target_FPR_0.1%   |               0.945341 |    0.97758 |    0.83148 |            0.99998 |         0.97707 |            0.50921 |                     0.99898 |   0.102 |   2.293 |                10.2 | 1522243 |   38 | 37066 | 35725 |   0.99706 |  0.99993 |            0.90622 |            0.94129 |            0.98047 |            0.97875 |

## 2. Saf C99 vs. ONNX Runtime Doğrulaması (1.59M Kayıt)

| model             | operating_point   |   c_fpr_% |   onnx_fpr_% |   c_recall_% |   onnx_recall_% |   c_vs_onnx_agree_% |   max_prob_diff |
|:------------------|:------------------|----------:|-------------:|-------------:|----------------:|--------------------:|----------------:|
| top15_l0_baseline | Target_FPR_0.5%   |     0.712 |        0.712 |       98.657 |          98.657 |                 100 |        0.035987 |
| top15_l0_baseline | Target_FPR_0.2%   |     0.248 |        0.248 |       98.493 |          98.493 |                 100 |        0.035987 |
| top15_l0_baseline | Target_FPR_0.1%   |     0.146 |        0.146 |       98.408 |          98.408 |                 100 |        0.035987 |
| top15_l1_balanced | Target_FPR_0.5%   |     0.447 |        0.447 |       98.52  |          98.52  |                 100 |        0.069997 |
| top15_l1_balanced | Target_FPR_0.2%   |     0.175 |        0.175 |       98.366 |          98.366 |                 100 |        0.069997 |
| top15_l1_balanced | Target_FPR_0.1%   |     0.111 |        0.111 |       98.274 |          98.274 |                 100 |        0.069997 |
| top10_l2_light    | Target_FPR_0.5%   |     0.423 |        0.423 |       98.367 |          98.367 |                 100 |        0.005807 |
| top10_l2_light    | Target_FPR_0.2%   |     0.146 |        0.146 |       97.994 |          97.994 |                 100 |        0.005807 |
| top10_l2_light    | Target_FPR_0.1%   |     0.102 |        0.102 |       97.707 |          97.707 |                 100 |        0.005807 |

## 3. Gerçekçi Canlı Normal Trafik Stres Testi (`%100 / %99 / %95 Normal`)

| model             | role                   | operating_point   |   locked_threshold | scenario              |   n_benign |   n_attack |   realized_fpr_% |   fp_per_10k_normal |   realized_precision |   realized_recall |   realized_f1_attack |   fp |   avg_tp |   avg_fn |
|:------------------|:-----------------------|:------------------|-------------------:|:----------------------|-----------:|-----------:|-----------------:|--------------------:|---------------------:|------------------:|---------------------:|-----:|---------:|---------:|
| top15_l0_baseline | Referans (Budanmamis)  | Target_FPR_0.2%   |           0.84289  | 100%_Normal_0%_Attack |      37104 |          0 |           0.248  |               24.8  |            nan       |         nan       |            nan       |   92 |      0   |      0   |
| top15_l0_baseline | Referans (Budanmamis)  | Target_FPR_0.2%   |           0.84289  | 99%_Normal_1%_Attack  |      37104 |        375 |           0.248  |               24.8  |              0.80079 |           0.98622 |              0.88388 |   92 |    369.8 |      5.2 |
| top15_l0_baseline | Referans (Budanmamis)  | Target_FPR_0.2%   |           0.84289  | 95%_Normal_5%_Attack  |      37104 |       1953 |           0.248  |               24.8  |              0.95437 |           0.98519 |              0.96953 |   92 |   1924.1 |     28.9 |
| top15_l0_baseline | Referans (Budanmamis)  | Target_FPR_0.1%   |           0.88668  | 100%_Normal_0%_Attack |      37104 |          0 |           0.1455 |               14.55 |            nan       |         nan       |            nan       |   54 |      0   |      0   |
| top15_l0_baseline | Referans (Budanmamis)  | Target_FPR_0.1%   |           0.88668  | 99%_Normal_1%_Attack  |      37104 |        375 |           0.1455 |               14.55 |              0.8725  |           0.98542 |              0.92552 |   54 |    369.5 |      5.5 |
| top15_l0_baseline | Referans (Budanmamis)  | Target_FPR_0.1%   |           0.88668  | 95%_Normal_5%_Attack  |      37104 |       1953 |           0.1455 |               14.55 |              0.97268 |           0.98443 |              0.97852 |   54 |   1922.6 |     30.4 |
| top15_l1_balanced | Ana Onerilen Model     | Target_FPR_0.2%   |           0.857909 | 100%_Normal_0%_Attack |      37104 |          0 |           0.1752 |               17.52 |            nan       |         nan       |            nan       |   65 |      0   |      0   |
| top15_l1_balanced | Ana Onerilen Model     | Target_FPR_0.2%   |           0.857909 | 99%_Normal_1%_Attack  |      37104 |        375 |           0.1752 |               17.52 |              0.85039 |           0.98524 |              0.91286 |   65 |    369.5 |      5.5 |
| top15_l1_balanced | Ana Onerilen Model     | Target_FPR_0.2%   |           0.857909 | 95%_Normal_5%_Attack  |      37104 |       1953 |           0.1752 |               17.52 |              0.96728 |           0.98401 |              0.97557 |   65 |   1921.8 |     31.2 |
| top15_l1_balanced | Ana Onerilen Model     | Target_FPR_0.1%   |           0.915133 | 100%_Normal_0%_Attack |      37104 |          0 |           0.1105 |               11.05 |            nan       |         nan       |            nan       |   41 |      0   |      0   |
| top15_l1_balanced | Ana Onerilen Model     | Target_FPR_0.1%   |           0.915133 | 99%_Normal_1%_Attack  |      37104 |        375 |           0.1105 |               11.05 |              0.90004 |           0.98444 |              0.94035 |   41 |    369.2 |      5.8 |
| top15_l1_balanced | Ana Onerilen Model     | Target_FPR_0.1%   |           0.915133 | 95%_Normal_5%_Attack  |      37104 |       1953 |           0.1105 |               11.05 |              0.97909 |           0.983   |              0.98104 |   41 |   1919.8 |     33.2 |
| top10_l2_light    | Ultra-Hafif Alternatif | Target_FPR_0.2%   |           0.891319 | 100%_Normal_0%_Attack |      37104 |          0 |           0.1455 |               14.55 |            nan       |         nan       |            nan       |   54 |      0   |      0   |
| top10_l2_light    | Ultra-Hafif Alternatif | Target_FPR_0.2%   |           0.891319 | 99%_Normal_1%_Attack  |      37104 |        375 |           0.1455 |               14.55 |              0.87208 |           0.98178 |              0.92368 |   54 |    368.2 |      6.8 |
| top10_l2_light    | Ultra-Hafif Alternatif | Target_FPR_0.2%   |           0.891319 | 95%_Normal_5%_Attack  |      37104 |       1953 |           0.1455 |               14.55 |              0.97256 |           0.98005 |              0.97629 |   54 |   1914   |     39   |
| top10_l2_light    | Ultra-Hafif Alternatif | Target_FPR_0.1%   |           0.945341 | 100%_Normal_0%_Attack |      37104 |          0 |           0.1024 |               10.24 |            nan       |         nan       |            nan       |   38 |      0   |      0   |
| top10_l2_light    | Ultra-Hafif Alternatif | Target_FPR_0.1%   |           0.945341 | 99%_Normal_1%_Attack  |      37104 |        375 |           0.1024 |               10.24 |              0.90622 |           0.9792  |              0.94129 |   38 |    367.2 |      7.8 |
| top10_l2_light    | Ultra-Hafif Alternatif | Target_FPR_0.1%   |           0.945341 | 95%_Normal_5%_Attack  |      37104 |       1953 |           0.1024 |               10.24 |              0.98047 |           0.97704 |              0.97875 |   38 |   1908.2 |     44.8 |

## 4. 7 Saldırı Ailesi Final Yakalama Oranları (`Target_FPR_0.1%`)

| category   |   support |   top10_l2_light |   top15_l0_baseline |   top15_l1_balanced |
|:-----------|----------:|-----------------:|--------------------:|--------------------:|
| Benign     |     37104 |           99.898 |              99.854 |              99.889 |
| BruteForce |       466 |            3.004 |               8.798 |               6.867 |
| DDoS       |   1152851 |           99.774 |              99.969 |              99.938 |
| DoS        |    274173 |           99.597 |              99.963 |              99.904 |
| Mirai      |     89359 |           99.382 |              99.935 |              99.877 |
| Recon      |     23641 |           15.689 |              24.013 |              20.722 |
| Spoofing   |     16596 |           38.497 |              69.529 |              65.245 |
| Web        |       882 |            1.701 |               5.556 |               2.381 |

## 5. Dosya Bazlı Kararlılık (`Merged09` vs `Merged10`)

| model             | operating_point   | test_file    |   n_samples |   accuracy |   attack_recall |   fpr_% |   fp |    fn |   roc_auc |
|:------------------|:------------------|:-------------|------------:|-----------:|----------------:|--------:|-----:|------:|----------:|
| top15_l0_baseline | Target_FPR_0.2%   | Merged09.csv |      678882 |    0.98542 |         0.98513 |   0.233 |   37 |  9859 |   0.99806 |
| top15_l0_baseline | Target_FPR_0.2%   | Merged10.csv |      916190 |    0.98507 |         0.98478 |   0.259 |   55 | 13626 |   0.99801 |
| top15_l0_baseline | Target_FPR_0.1%   | Merged09.csv |      678882 |    0.98466 |         0.98433 |   0.132 |   21 | 10392 |   0.99806 |
| top15_l0_baseline | Target_FPR_0.1%   | Merged10.csv |      916190 |    0.98424 |         0.9839  |   0.156 |   33 | 14407 |   0.99801 |
| top15_l1_balanced | Target_FPR_0.2%   | Merged09.csv |      678882 |    0.98423 |         0.98389 |   0.176 |   28 | 10679 |   0.99773 |
| top15_l1_balanced | Target_FPR_0.2%   | Merged10.csv |      916190 |    0.98383 |         0.98349 |   0.175 |   37 | 14775 |   0.99768 |
| top15_l1_balanced | Target_FPR_0.1%   | Merged09.csv |      678882 |    0.98333 |         0.98296 |   0.101 |   16 | 11300 |   0.99773 |
| top15_l1_balanced | Target_FPR_0.1%   | Merged10.csv |      916190 |    0.98295 |         0.98257 |   0.118 |   25 | 15598 |   0.99768 |
| top10_l2_light    | Target_FPR_0.2%   | Merged09.csv |      678882 |    0.98056 |         0.98013 |   0.132 |   21 | 13175 |   0.99711 |
| top10_l2_light    | Target_FPR_0.2%   | Merged10.csv |      916190 |    0.98023 |         0.9798  |   0.156 |   33 | 18076 |   0.99702 |
| top10_l2_light    | Target_FPR_0.1%   | Merged09.csv |      678882 |    0.97779 |         0.97728 |   0.094 |   15 | 15066 |   0.99711 |
| top10_l2_light    | Target_FPR_0.1%   | Merged10.csv |      916190 |    0.97743 |         0.97692 |   0.109 |   23 | 20659 |   0.99702 |

## 6. Final Saf C99 Donanım Özeti

| model             | role                   |   n_features |   onnx_size_kb |   c_dll_size_kb |   c_exe_size_kb |   c_rss_peak_mb |   c_lat_p50_us |   c_lat_p95_us |   c_lat_p99_us |   c_single_core_pps |   c_full_1_59m_sec |   onnx_full_1_59m_sec |
|:------------------|:-----------------------|-------------:|---------------:|----------------:|----------------:|----------------:|---------------:|---------------:|---------------:|--------------------:|-------------------:|----------------------:|
| top15_l0_baseline | Referans (Budanmamis)  |           15 |         673.58 |          487.38 |          508    |           4.156 |            0.5 |            3.6 |            5.6 |    973687           |              1.417 |                 5.476 |
| top15_l1_balanced | Ana Onerilen Model     |           15 |         174.97 |          191.58 |          212.19 |           3.871 |            0.2 |            0.8 |            1.3 |         4.86074e+06 |              0.351 |                 2.673 |
| top10_l2_light    | Ultra-Hafif Alternatif |           10 |          52.63 |          101.7  |          123.31 |           3.75  |            0.1 |            0.3 |            0.5 |         1.64935e+07 |              0.111 |                 1.391 |