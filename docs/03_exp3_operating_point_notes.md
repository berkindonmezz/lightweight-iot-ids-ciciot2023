# Exp 3 — Binary LightGBM Eşik Optimizasyonu ve Canlı Normal Trafik Stres Testi

- **Özellik Seti (24 Özellik):** `Header_Length, Protocol Type, Time_To_Live, Rate, fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ece_flag_number, HTTP, HTTPS, DNS, SSH, TCP, UDP, DHCP, ARP, ICMP, Min, Max, AVG, Std, IAT`
- **Bölme:** Train=`Merged01–04`, Eşik Kalibrasyonu=`Merged07.csv`, Değerlendirme=`Merged08.csv`, Kilitli Test=`['Merged09.csv', 'Merged10.csv']`

## 1. FPR Bütçelerine Göre Çalışma Noktaları (`Merged08`)

| model                         | operating_point   |   threshold_from_m07 |   accuracy |   f1_macro |   attack_precision |   attack_recall |   benign_recall |     fpr |   fp_per_10k_normal |   fp |    fn |   roc_auc |   pr_auc |
|:------------------------------|:------------------|---------------------:|-----------:|-----------:|-------------------:|----------------:|----------------:|--------:|--------------------:|-----:|------:|----------:|---------:|
| LGBM_T1_25pctBenign_None      | Default_0.50      |             0.5      |    0.99238 |    0.91742 |            0.99663 |         0.99557 |         0.85759 | 0.14241 |              1424.1 | 2345 |  3085 |   0.99786 |  0.99995 |
| LGBM_T1_25pctBenign_None      | Target_FPR_1.0%   |             0.903597 |    0.98716 |    0.88721 |            0.99976 |         0.98709 |         0.99016 | 0.00984 |                98.4 |  162 |  8980 |   0.99786 |  0.99995 |
| LGBM_T1_25pctBenign_None      | Target_FPR_0.5%   |             0.929409 |    0.98603 |    0.87998 |            0.99988 |         0.98582 |         0.9952  | 0.0048  |                48   |   79 |  9869 |   0.99786 |  0.99995 |
| LGBM_T1_25pctBenign_None      | Target_FPR_0.2%   |             0.965532 |    0.98442 |    0.86988 |            0.99997 |         0.98409 |         0.99866 | 0.00134 |                13.4 |   22 | 11071 |   0.99786 |  0.99995 |
| LGBM_T1_25pctBenign_None      | Target_FPR_0.1%   |             0.983737 |    0.98308 |    0.86158 |            0.99998 |         0.9827  |         0.99909 | 0.00091 |                 9.1 |   15 | 12038 |   0.99786 |  0.99995 |
| LGBM_T2_100pctBenign_None     | Default_0.50      |             0.5      |    0.99016 |    0.9076  |            0.99931 |         0.99062 |         0.97091 | 0.02909 |               290.9 |  479 |  6528 |   0.99811 |  0.99996 |
| LGBM_T2_100pctBenign_None     | Target_FPR_1.0%   |             0.657415 |    0.988   |    0.89306 |            0.99975 |         0.98797 |         0.98955 | 0.01045 |               104.5 |  172 |  8372 |   0.99811 |  0.99996 |
| LGBM_T2_100pctBenign_None     | Target_FPR_0.5%   |             0.739057 |    0.9867  |    0.88446 |            0.99988 |         0.9865  |         0.9952  | 0.0048  |                48   |   79 |  9394 |   0.99811 |  0.99996 |
| LGBM_T2_100pctBenign_None     | Target_FPR_0.2%   |             0.849165 |    0.98558 |    0.87727 |            0.99995 |         0.98529 |         0.99812 | 0.00188 |                18.8 |   31 | 10238 |   0.99811 |  0.99996 |
| LGBM_T2_100pctBenign_None     | Target_FPR_0.1%   |             0.880404 |    0.98505 |    0.87387 |            0.99998 |         0.98471 |         0.99909 | 0.00091 |                 9.1 |   15 | 10636 |   0.99811 |  0.99996 |
| LGBM_T2_100pctBenign_Balanced | Default_0.50      |             0.5      |    0.98532 |    0.87558 |            0.99997 |         0.985   |         0.9986  | 0.0014  |                14   |   23 | 10435 |   0.99804 |  0.99995 |
| LGBM_T2_100pctBenign_Balanced | Target_FPR_1.0%   |             0.181946 |    0.98817 |    0.89427 |            0.99975 |         0.98814 |         0.98974 | 0.01026 |               102.6 |  169 |  8255 |   0.99804 |  0.99995 |
| LGBM_T2_100pctBenign_Balanced | Target_FPR_0.5%   |             0.231361 |    0.9872  |    0.88783 |            0.99987 |         0.98702 |         0.99472 | 0.00528 |                52.8 |   87 |  9032 |   0.99804 |  0.99995 |
| LGBM_T2_100pctBenign_Balanced | Target_FPR_0.2%   |             0.392288 |    0.98595 |    0.87968 |            0.99995 |         0.98567 |         0.99781 | 0.00219 |                21.9 |   36 |  9969 |   0.99804 |  0.99995 |
| LGBM_T2_100pctBenign_Balanced | Target_FPR_0.1%   |             0.537315 |    0.98514 |    0.87445 |            0.99997 |         0.98482 |         0.99866 | 0.00134 |                13.4 |   22 | 10561 |   0.99804 |  0.99995 |

## 2. Canlı Normal Trafik Stres Testi (LGBM_T2_100pctBenign_Balanced)

| scenario              |   n_benign |   n_attack |   realized_fpr |   fp_per_10k_normal |   realized_precision |   realized_recall |   realized_f1_attack |   avg_fp |   avg_tp |   avg_fn | model                         | operating_point   |   threshold |
|:----------------------|-----------:|-----------:|---------------:|--------------------:|---------------------:|------------------:|---------------------:|---------:|---------:|---------:|:------------------------------|:------------------|------------:|
| 100%_Normal_0%_Attack |      16467 |          0 |        0.01026 |               102.6 |            nan       |         nan       |            nan       |      169 |      0   |      0   | LGBM_T2_100pctBenign_Balanced | Target_FPR_1.0%   |    0.181946 |
| 99%_Normal_1%_Attack  |      16467 |        166 |        0.01026 |               102.6 |              0.49241 |           0.98765 |              0.65717 |      169 |    163.9 |      2   | LGBM_T2_100pctBenign_Balanced | Target_FPR_1.0%   |    0.181946 |
| 95%_Normal_5%_Attack  |      16467 |        867 |        0.01026 |               102.6 |              0.83527 |           0.98841 |              0.90541 |      169 |    857   |     10.1 | LGBM_T2_100pctBenign_Balanced | Target_FPR_1.0%   |    0.181946 |
| 100%_Normal_0%_Attack |      16467 |          0 |        0.00528 |                52.8 |            nan       |         nan       |            nan       |       87 |      0   |      0   | LGBM_T2_100pctBenign_Balanced | Target_FPR_0.5%   |    0.231361 |
| 99%_Normal_1%_Attack  |      16467 |        166 |        0.00528 |                52.8 |              0.65317 |           0.98705 |              0.78612 |       87 |    163.8 |      2.1 | LGBM_T2_100pctBenign_Balanced | Target_FPR_0.5%   |    0.231361 |
| 95%_Normal_5%_Attack  |      16467 |        867 |        0.00528 |                52.8 |              0.90775 |           0.98749 |              0.94594 |       87 |    856.1 |     10.8 | LGBM_T2_100pctBenign_Balanced | Target_FPR_0.5%   |    0.231361 |
| 100%_Normal_0%_Attack |      16467 |          0 |        0.00219 |                21.9 |            nan       |         nan       |            nan       |       36 |      0   |      0   | LGBM_T2_100pctBenign_Balanced | Target_FPR_0.2%   |    0.392288 |
| 99%_Normal_1%_Attack  |      16467 |        166 |        0.00219 |                21.9 |              0.81967 |           0.98584 |              0.8951  |       36 |    163.7 |      2.4 | LGBM_T2_100pctBenign_Balanced | Target_FPR_0.2%   |    0.392288 |
| 95%_Normal_5%_Attack  |      16467 |        867 |        0.00219 |                21.9 |              0.9596  |           0.98633 |              0.97278 |       36 |    855.1 |     11.8 | LGBM_T2_100pctBenign_Balanced | Target_FPR_0.2%   |    0.392288 |
| 100%_Normal_0%_Attack |      16467 |          0 |        0.00134 |                13.4 |            nan       |         nan       |            nan       |       22 |      0   |      0   | LGBM_T2_100pctBenign_Balanced | Target_FPR_0.1%   |    0.537315 |
| 99%_Normal_1%_Attack  |      16467 |        166 |        0.00134 |                13.4 |              0.88146 |           0.98554 |              0.93058 |       22 |    163.6 |      2.4 | LGBM_T2_100pctBenign_Balanced | Target_FPR_0.1%   |    0.537315 |
| 95%_Normal_5%_Attack  |      16467 |        867 |        0.00134 |                13.4 |              0.97489 |           0.98524 |              0.98003 |       22 |    854.2 |     12.8 | LGBM_T2_100pctBenign_Balanced | Target_FPR_0.1%   |    0.537315 |

## 3. Saldırı Ailesi Bazlı Yakalama Oranları (LGBM_T2_100pctBenign_Balanced)

| operating_point   | category   |   support |   correct_or_detected |   fp_or_missed |   rate_pct | metric_note                  |
|:------------------|:-----------|----------:|----------------------:|---------------:|-----------:|:-----------------------------|
| Target_FPR_1.0%   | Benign     |     16467 |                 16298 |            169 |      98.97 | Benign Specificity (1 - FPR) |
| Target_FPR_1.0%   | BruteForce |       220 |                    65 |            155 |      29.55 | Attack Recall                |
| Target_FPR_1.0%   | DDoS       |    514947 |                514905 |             42 |      99.99 | Attack Recall                |
| Target_FPR_1.0%   | DoS        |    122605 |                122593 |             12 |      99.99 | Attack Recall                |
| Target_FPR_1.0%   | Mirai      |     40069 |                 40065 |              4 |      99.99 | Attack Recall                |
| Target_FPR_1.0%   | Recon      |     10260 |                  3871 |           6389 |      37.73 | Attack Recall                |
| Target_FPR_1.0%   | Spoofing   |      7295 |                  5918 |           1377 |      81.12 | Attack Recall                |
| Target_FPR_1.0%   | Web        |       357 |                    81 |            276 |      22.69 | Attack Recall                |
| Target_FPR_0.5%   | Benign     |     16467 |                 16380 |             87 |      99.47 | Benign Specificity (1 - FPR) |
| Target_FPR_0.5%   | BruteForce |       220 |                    61 |            159 |      27.73 | Attack Recall                |
| Target_FPR_0.5%   | DDoS       |    514947 |                514889 |             58 |      99.99 | Attack Recall                |
| Target_FPR_0.5%   | DoS        |    122605 |                122588 |             17 |      99.99 | Attack Recall                |
| Target_FPR_0.5%   | Mirai      |     40069 |                 40063 |              6 |      99.99 | Attack Recall                |
| Target_FPR_0.5%   | Recon      |     10260 |                  3473 |           6787 |      33.85 | Attack Recall                |
| Target_FPR_0.5%   | Spoofing   |      7295 |                  5586 |           1709 |      76.57 | Attack Recall                |
| Target_FPR_0.5%   | Web        |       357 |                    61 |            296 |      17.09 | Attack Recall                |
| Target_FPR_0.2%   | Benign     |     16467 |                 16431 |             36 |      99.78 | Benign Specificity (1 - FPR) |
| Target_FPR_0.2%   | BruteForce |       220 |                    54 |            166 |      24.55 | Attack Recall                |
| Target_FPR_0.2%   | DDoS       |    514947 |                514844 |            103 |      99.98 | Attack Recall                |
| Target_FPR_0.2%   | DoS        |    122605 |                122573 |             32 |      99.97 | Attack Recall                |
| Target_FPR_0.2%   | Mirai      |     40069 |                 40060 |              9 |      99.98 | Attack Recall                |
| Target_FPR_0.2%   | Recon      |     10260 |                  2875 |           7385 |      28.02 | Attack Recall                |
| Target_FPR_0.2%   | Spoofing   |      7295 |                  5342 |           1953 |      73.23 | Attack Recall                |
| Target_FPR_0.2%   | Web        |       357 |                    36 |            321 |      10.08 | Attack Recall                |
| Target_FPR_0.1%   | Benign     |     16467 |                 16445 |             22 |      99.87 | Benign Specificity (1 - FPR) |
| Target_FPR_0.1%   | BruteForce |       220 |                    48 |            172 |      21.82 | Attack Recall                |
| Target_FPR_0.1%   | DDoS       |    514947 |                514811 |            136 |      99.97 | Attack Recall                |
| Target_FPR_0.1%   | DoS        |    122605 |                122566 |             39 |      99.97 | Attack Recall                |
| Target_FPR_0.1%   | Mirai      |     40069 |                 40053 |             16 |      99.96 | Attack Recall                |
| Target_FPR_0.1%   | Recon      |     10260 |                  2581 |           7679 |      25.16 | Attack Recall                |
| Target_FPR_0.1%   | Spoofing   |      7295 |                  5113 |           2182 |      70.09 | Attack Recall                |
| Target_FPR_0.1%   | Web        |       357 |                    20 |            337 |       5.6  | Attack Recall                |