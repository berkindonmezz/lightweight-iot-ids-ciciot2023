# CICIoT2023 Özellik Denetimi, Eleme Kararları ve Model Karşılaştırma Notları

- **Eğitim Seti (File-Disjoint):** `Merged01–04` (708,703 satır)
- **Doğrulama Seti (File-Disjoint):** `Merged07` (702,212 satır)
- **Kilitli Test Seti:** `Merged09–10` (Bu aşamada hiç açılmamıştır)

## 1. Çıkarılan Özellikler ve Gerekçeleri

### 1.1. Pencere-Boyutu Artefaktları (6 Özellik — Kesin Çıkarıldı)
CICIoT2023'te normal/düşük hacimli sınıflar 10 paketlik, DDoS/DoS/Mirai sınıfları 100 paketlik pencerelerle üretilmiştir. Canlı trafikte sabit pencere kullanılacağı için bu sütunlar ezber/sızıntı kaynağıdır:
- **`Number`**: Penceredeki paket sayısı; veri setinde 10 vs 100 pencere kuralını sızdırır.
- **`Tot sum`**: Penceredeki toplam bayt (Number * AVG); pencere boyutuna bağımlı.
- **`ack_count`**: Penceredeki toplam ACK sayısı; pencere boyutuna (10 vs 100) bağımlı.
- **`syn_count`**: Penceredeki toplam SYN sayısı; pencere boyutuna (10 vs 100) bağımlı.
- **`fin_count`**: Penceredeki toplam FIN sayısı; pencere boyutuna (10 vs 100) bağımlı.
- **`rst_count`**: Penceredeki toplam RST sayısı; pencere boyutuna (10 vs 100) bağımlı.

### 1.2. Matematiksel Kopya / Kolineer Sütunlar (3 Özellik — Kesin Çıkarıldı)
- **`Tot size`**: AVG sütunu ile birebir aynı değerleri taşır (tam kopya).
- **`IPv`**: IPv4 çerçeve oranı; ARP ile |r|=1.0 tam doğrusal bağımlı (kopya).
- **`LLC`**: LLC katmanı oranı; ARP ve IPv ile |r|=1.0 tam doğrusal bağımlı (kopya).

### 1.3. Sıfır / İhmal Edilebilir Katkılı Özellikler (5 Özellik — Deneysel Eleme)
- **`cwr_flag_number`**: Sıfır-dışı oran=%0.035, Bin Gain=0.00000, MC8 Gain=0.00000, Val PermImp(Bin/MC8)=0.00000/0.00000 — TCP Congestion Window Reduced bayrağı oranı (pratikte çok nadir).
- **`Telnet`**: Sıfır-dışı oran=%0.077, Bin Gain=0.00000, MC8 Gain=0.00000, Val PermImp(Bin/MC8)=0.00000/0.00000 — Penceredeki Telnet trafiği oranı (Mirai/brute-force potansiyeli, veri setinde nadir).
- **`SMTP`**: Sıfır-dışı oran=%0.075, Bin Gain=0.00000, MC8 Gain=0.00000, Val PermImp(Bin/MC8)=0.00000/0.00000 — Penceredeki SMTP e-posta trafiği oranı.
- **`IRC`**: Sıfır-dışı oran=%0.090, Bin Gain=0.00000, MC8 Gain=0.00000, Val PermImp(Bin/MC8)=0.00000/0.00000 — Penceredeki IRC trafiği oranı.
- **`IGMP`**: Sıfır-dışı oran=%0.063, Bin Gain=0.00000, MC8 Gain=0.00000, Val PermImp(Bin/MC8)=0.00000/0.00000 — Penceredeki IGMP multicast kontrol paketi oranı.

## 2. Kalan Özellikler ve Ne İşe Yaradıkları (25 Özellik)

| Özellik | Kategori | Sıfır-Dışı (%) | Bin Gain | MC8 Gain | Bin PermImp | MC8 PermImp | Canlı Trafik | Açıklama |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `AVG` | Paket Boyutu | %100.00 | 0.0005 | 0.0024 | 0.21641 | 0.18481 | Kolay (ortalama boyut) | Penceredeki ortalama paket boyutu (bayt). |
| `HTTPS` | Uygulama/Port | %14.39 | 0.0022 | 0.0006 | 0.29597 | 0.06847 | Kolay (Port 443 kontrolu) | Penceredeki HTTPS/TLS trafiği oranı (IoT cihaz normal iletişimi ile yüksek ilişkili). |
| `Header_Length` | Akis/Baslik | %84.11 | 0.0031 | 0.0097 | 0.16480 | 0.13514 | Kolay (IP/TCP/UDP header) | Penceredeki paketlerin ortalama başlık uzunluğu (bayt). |
| `ack_flag_number` | TCP Bayrak | %25.59 | 0.0127 | 0.0098 | 0.11313 | 0.10233 | Kolay (TCP flags) | Penceredeki TCP ACK bayrağı oranı (kurulu oturum / ACK flood davranışı). |
| `Max` | Paket Boyutu | %100.00 | 0.0031 | 0.0706 | 0.04123 | 0.16365 | Kolay (max paket boyutu) | Penceredeki en büyük paket boyutu (bayt). |
| `IAT` | Akis/Zaman | %100.00 | 0.1375 | 0.7614 | 0.08142 | 0.11883 | Kolay (paketler arasi sure) | Paketler arası ortalama varış süresi (Inter-Arrival Time); en güçlü davranışsal sinyal. |
| `Std` | Paket Boyutu | %26.27 | 0.0006 | 0.0009 | 0.01405 | 0.13587 | Kolay (boyut std) | Penceredeki paket boyutlarının standart sapması (flood trafiğinde ~0'a iner). |
| `Variance` | Paket Boyutu | %26.27 | 0.0000 | 0.0018 | 0.00126 | 0.13395 | Kısmen Kopya (Std^2) | Paket boyutu varyansı (Std'nin karesi); canlı trafikte ikisinden biri yeterlidir. |
| `syn_flag_number` | TCP Bayrak | %26.12 | 0.0004 | 0.0025 | 0.05787 | 0.04250 | Kolay (TCP flags) | Penceredeki TCP SYN bayrağı oranı (SYN flood ve port tarama göstergesi). |
| `Min` | Paket Boyutu | %100.00 | 0.0001 | 0.0090 | 0.00599 | 0.08257 | Kolay (min paket boyutu) | Penceredeki en küçük paket boyutu (bayt). |
| `Rate` | Akis/Hiz | %100.00 | 0.7051 | 0.0597 | 0.01514 | 0.04654 | Kolay (paket_sayisi / sure) | Penceredeki paket iletim hızı (paket/sn); DoS/DDoS ve tarama ayrımında temel sinyal. |
| `fin_flag_number` | TCP Bayrak | %11.41 | 0.0562 | 0.0148 | 0.01874 | 0.04116 | Kolay (TCP flags) | Penceredeki TCP FIN bayrağı oranı (bağlantı sonlandırma yoğunluğu). |
| `psh_flag_number` | TCP Bayrak | %17.21 | 0.0116 | 0.0090 | 0.01194 | 0.04789 | Kolay (TCP flags) | Penceredeki TCP PSH bayrağı oranı (anlık veri itme / PSH-ACK flood). |
| `UDP` | Protokol | %31.25 | 0.0000 | 0.0032 | 0.00105 | 0.05701 | Kolay (L4 protokol) | Penceredeki UDP paket oranı. |
| `Time_To_Live` | Akis/Baslik | %100.00 | 0.0069 | 0.0044 | 0.01890 | 0.03167 | Kolay (IP TTL alani) | Ortalama IP TTL değeri; kaynak işletim sistemi ve yönlendirme/spoofing izi taşır. |
| `TCP` | Protokol | %65.03 | 0.0000 | 0.0002 | 0.00016 | 0.05029 | Kolay (L4 protokol) | Penceredeki TCP paket oranı. |
| `Protocol Type` | Protokol | %99.97 | 0.0000 | 0.0005 | 0.00000 | 0.04567 | Kolay (IP proto alani) | IP protokol numarası (TCP=6, UDP=17, ICMP=1 vb.). |
| `rst_flag_number` | TCP Bayrak | %12.48 | 0.0024 | 0.0234 | 0.01960 | 0.02562 | Kolay (TCP flags) | Penceredeki TCP RST bayrağı oranı (reddedilen/sıfırlanan bağlantılar). |
| `ICMP` | Protokol | %20.98 | 0.0000 | 0.0127 | -0.00114 | 0.03626 | Kolay (IP proto=1) | Penceredeki ICMP paket oranı (ICMP flood / ping sweep). |
| `HTTP` | Uygulama/Port | %7.73 | 0.0575 | 0.0021 | 0.00036 | 0.01666 | Kolay (Port 80 kontrolu) | Penceredeki HTTP trafiği oranı (port tabanlı uygulama göstergesi). |
| `ARP` | Protokol | %9.81 | 0.0000 | 0.0000 | 0.00614 | 0.00594 | Kolay (Ethernet type) | Penceredeki ARP çerçevesi oranı (MITM-ARP Spoofing ve keşif için kritik). |
| `SSH` | Uygulama/Port | %0.31 | 0.0000 | 0.0000 | 0.00018 | 0.00409 | Kolay (Port 22 kontrolu) | Penceredeki SSH trafiği oranı (Dictionary Brute Force için kritik). |
| `DHCP` | Uygulama/Port | %0.55 | 0.0000 | 0.0000 | 0.00010 | 0.00030 | Kolay (Port 67/68) | Penceredeki DHCP trafiği oranı. |
| `DNS` | Uygulama/Port | %6.08 | 0.0000 | 0.0013 | 0.00017 | 0.00019 | Kolay (Port 53 kontrolu) | Penceredeki DNS trafiği oranı (DNS spoofing / normal ad çözümleme). |
| `ece_flag_number` | TCP Bayrak | %0.08 | 0.0000 | 0.0000 | 0.00000 | 0.00013 | Kolay (TCP flags) | TCP ECN-Echo tıkanıklık bayrağı oranı (pratikte çok nadir). |

## 3. Özellik Eleme ve Canlı Trafik Ablation Kıyaslaması (LightGBM, Val)

| feature_set                |   n_features |   bin_acc |   bin_atk_prec |   bin_atk_rec |   bin_ben_prec |   bin_ben_rec |   bin_fpr |   bin_fnr |   bin_fp |   bin_fn |   bin_f1_macro |   bin_roc_auc |   bin_pr_auc |   mc8_acc |   mc8_bal_acc |   mc8_prec_macro |   mc8_rec_macro |   mc8_f1_macro |   mc8_f1_weighted |   mc8_ovr_roc |
|:---------------------------|-------------:|----------:|---------------:|--------------:|---------------:|--------------:|----------:|----------:|---------:|---------:|---------------:|--------------:|-------------:|----------:|--------------:|-----------------:|----------------:|---------------:|------------------:|--------------:|
| S1_Clean_30                |           30 |   0.99119 |        0.996   |       0.99497 |        0.79561 |       0.83042 |   0.16958 |   0.00503 |     2741 |     3448 |        0.90406 |       0.99701 |      0.99992 |   0.84387 |       0.59123 |          0.6052  |         0.59123 |        0.5836  |           0.82557 |       0.87055 |
| S2_Pruned_25               |           25 |   0.99198 |        0.99604 |       0.99575 |        0.82185 |       0.83202 |   0.16798 |   0.00425 |     2715 |     2915 |        0.9114  |       0.9978  |      0.99994 |   0.80829 |       0.57057 |          0.55704 |         0.57057 |        0.55588 |           0.80126 |       0.87817 |
| S3_Pruned_NoVariance_24    |           24 |   0.99198 |        0.99599 |       0.9958  |        0.82329 |       0.82986 |   0.17014 |   0.0042  |     2750 |     2879 |        0.91123 |       0.99784 |      0.99994 |   0.82173 |       0.59188 |          0.54124 |         0.59188 |        0.55393 |           0.81245 |       0.88684 |
| S4_PureFlow_NoPortProto_20 |           20 |   0.99165 |        0.99591 |       0.99554 |        0.81355 |       0.82664 |   0.17336 |   0.00446 |     2802 |     3062 |        0.90789 |       0.99763 |      0.99994 |   0.81187 |       0.58578 |          0.53225 |         0.58578 |        0.54476 |           0.80168 |       0.92048 |

## 4. Tam Metrikli Model Karşılaştırması — Binary (Val)

| model               | class_weight   |   train_sec |   train_f1_macro |   val_f1_macro |   overfit_gap_f1 |   accuracy |   attack_precision |   attack_recall |   benign_precision |   benign_recall |     fpr |     fnr |   fp |   fn |   roc_auc |   pr_auc |   tuned_thr |   tuned_f1_macro |   tuned_fpr |   tuned_attack_recall |
|:--------------------|:---------------|------------:|-----------------:|---------------:|-----------------:|-----------:|-------------------:|----------------:|-------------------:|----------------:|--------:|--------:|-----:|-----:|----------:|---------:|------------:|-----------------:|------------:|----------------------:|
| Random Forest       | None           |       40.53 |          0.99957 |        0.91372 |          0.08585 |    0.99229 |            0.99589 |         0.99622 |            0.83734 |         0.82553 | 0.17447 | 0.00378 | 2820 | 2592 |   0.99757 |  0.99993 |       0.575 |          0.91743 |     0.12417 |               0.995   |
| LightGBM            | None           |        6.6  |          0.91917 |        0.9114  |          0.00777 |    0.99198 |            0.99604 |         0.99575 |            0.82185 |         0.83202 | 0.16798 | 0.00425 | 2715 | 2915 |   0.9978  |  0.99994 |       0.625 |          0.91502 |     0.10901 |               0.99422 |
| Decision Tree       | None           |       11.26 |          0.9996  |        0.8804  |          0.1192  |    0.98923 |            0.99451 |         0.99447 |            0.76562 |         0.767   | 0.233   | 0.00553 | 3766 | 3795 |   0.88129 |  0.99444 |       0.05  |          0.8804  |     0.233   |               0.99447 |
| LightGBM (balanced) | balanced       |        6.72 |          0.88024 |        0.87854 |          0.0017  |    0.98587 |            0.99989 |         0.98565 |            0.62039 |         0.99536 | 0.00464 | 0.01435 |   75 | 9844 |   0.99786 |  0.99995 |       0.05  |          0.91516 |     0.07839 |               0.99324 |
| Logistic Regression | None           |        6.34 |          0.82985 |        0.83127 |         -0.00142 |    0.98539 |            0.99164 |         0.99343 |            0.69785 |         0.64431 | 0.35569 | 0.00657 | 5749 | 4509 |   0.99007 |  0.99953 |       0.7   |          0.84758 |     0.21574 |               0.98947 |

## 5. Tam Metrikli Model Karşılaştırması — 8-Sınıf (Val)

| model               | class_weight   |   train_sec |   train_f1_macro |   val_f1_macro |   overfit_gap_f1 |   accuracy |   balanced_accuracy |   precision_macro |   recall_macro |   f1_weighted |   benign_misclassified_rate |   roc_auc_ovr_macro |   pr_auc_ovr_macro |
|:--------------------|:---------------|------------:|-----------------:|---------------:|-----------------:|-----------:|--------------------:|------------------:|---------------:|--------------:|----------------------------:|--------------------:|-------------------:|
| LightGBM (balanced) | balanced       |       28.65 |          0.75279 |        0.64333 |          0.10946 |    0.75611 |             0.70498 |           0.63344 |        0.70498 |       0.78068 |                     0.19402 |             0.97585 |            0.70999 |
| Random Forest       | None           |       61.29 |          0.95799 |        0.64227 |          0.31572 |    0.84881 |             0.61867 |           0.78386 |        0.61867 |       0.83412 |                     0.12819 |             0.93941 |            0.6913  |
| Decision Tree       | None           |       11.75 |          0.97282 |        0.60268 |          0.37014 |    0.82334 |             0.6015  |           0.60532 |        0.6015  |       0.8169  |                     0.23214 |             0.78817 |            0.50215 |
| LightGBM            | None           |       29.17 |          0.62418 |        0.55588 |          0.0683  |    0.80829 |             0.57057 |           0.55704 |        0.57057 |       0.80126 |                     0.23498 |             0.87817 |            0.50586 |
| Logistic Regression | None           |       81.13 |          0.52337 |        0.51452 |          0.00885 |    0.82543 |             0.48856 |           0.64911 |        0.48856 |       0.7864  |                     0.20547 |             0.95861 |            0.56277 |