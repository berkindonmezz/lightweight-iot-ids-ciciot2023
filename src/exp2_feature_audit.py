import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    get_hardware_profile,
)
from src.config import (
    CLASS_8_ORDER,
    DOCS_DIR,
    FEATURES_33,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_FILES_LOCKED,
    TRAIN_FILES,
    WINDOW_DEPENDENT_COLS,
)
from src.preprocessing import build_feature_pipeline, load_csv_files

warnings.filterwarnings("ignore", category=UserWarning)

VAL_FILES_EXP2 = ["Merged07.csv"]
ALL_39_FEATURES = FEATURES_33 + WINDOW_DEPENDENT_COLS

EXACT_DUPLICATE_COLS = ["Tot size", "IPv", "LLC"]
PORT_DERIVED_PROTOCOL_COLS = ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "DHCP"]
CLEAN_30_FEATURES = [c for c in FEATURES_33 if c not in EXACT_DUPLICATE_COLS]

# Tüm 39 özellik için raporluk Türkçe sözlük ve canlı trafik uyumluluk notu
FEATURE_DICTIONARY = {
    "Header_Length": ("Akis/Baslik", "Kolay (IP/TCP/UDP header)", "Penceredeki paketlerin ortalama başlık uzunluğu (bayt)."),
    "Protocol Type": ("Protokol", "Kolay (IP proto alani)", "IP protokol numarası (TCP=6, UDP=17, ICMP=1 vb.)."),
    "Time_To_Live": ("Akis/Baslik", "Kolay (IP TTL alani)", "Ortalama IP TTL değeri; kaynak işletim sistemi ve yönlendirme/spoofing izi taşır."),
    "Rate": ("Akis/Hiz", "Kolay (paket_sayisi / sure)", "Penceredeki paket iletim hızı (paket/sn); DoS/DDoS ve tarama ayrımında temel sinyal."),
    "fin_flag_number": ("TCP Bayrak", "Kolay (TCP flags)", "Penceredeki TCP FIN bayrağı oranı (bağlantı sonlandırma yoğunluğu)."),
    "syn_flag_number": ("TCP Bayrak", "Kolay (TCP flags)", "Penceredeki TCP SYN bayrağı oranı (SYN flood ve port tarama göstergesi)."),
    "rst_flag_number": ("TCP Bayrak", "Kolay (TCP flags)", "Penceredeki TCP RST bayrağı oranı (reddedilen/sıfırlanan bağlantılar)."),
    "psh_flag_number": ("TCP Bayrak", "Kolay (TCP flags)", "Penceredeki TCP PSH bayrağı oranı (anlık veri itme / PSH-ACK flood)."),
    "ack_flag_number": ("TCP Bayrak", "Kolay (TCP flags)", "Penceredeki TCP ACK bayrağı oranı (kurulu oturum / ACK flood davranışı)."),
    "ece_flag_number": ("TCP Bayrak", "Kolay (TCP flags)", "TCP ECN-Echo tıkanıklık bayrağı oranı (pratikte çok nadir)."),
    "cwr_flag_number": ("TCP Bayrak", "Kolay (TCP flags)", "TCP Congestion Window Reduced bayrağı oranı (pratikte çok nadir)."),
    "HTTP": ("Uygulama/Port", "Kolay (Port 80 kontrolu)", "Penceredeki HTTP trafiği oranı (port tabanlı uygulama göstergesi)."),
    "HTTPS": ("Uygulama/Port", "Kolay (Port 443 kontrolu)", "Penceredeki HTTPS/TLS trafiği oranı (IoT cihaz normal iletişimi ile yüksek ilişkili)."),
    "DNS": ("Uygulama/Port", "Kolay (Port 53 kontrolu)", "Penceredeki DNS trafiği oranı (DNS spoofing / normal ad çözümleme)."),
    "Telnet": ("Uygulama/Port", "Kolay (Port 23 kontrolu)", "Penceredeki Telnet trafiği oranı (Mirai/brute-force potansiyeli, veri setinde nadir)."),
    "SMTP": ("Uygulama/Port", "Kolay (Port 25 kontrolu)", "Penceredeki SMTP e-posta trafiği oranı."),
    "SSH": ("Uygulama/Port", "Kolay (Port 22 kontrolu)", "Penceredeki SSH trafiği oranı (Dictionary Brute Force için kritik)."),
    "IRC": ("Uygulama/Port", "Kolay (Port 6667 kontrolu)", "Penceredeki IRC trafiği oranı."),
    "TCP": ("Protokol", "Kolay (L4 protokol)", "Penceredeki TCP paket oranı."),
    "UDP": ("Protokol", "Kolay (L4 protokol)", "Penceredeki UDP paket oranı."),
    "DHCP": ("Uygulama/Port", "Kolay (Port 67/68)", "Penceredeki DHCP trafiği oranı."),
    "ARP": ("Protokol", "Kolay (Ethernet type)", "Penceredeki ARP çerçevesi oranı (MITM-ARP Spoofing ve keşif için kritik)."),
    "ICMP": ("Protokol", "Kolay (IP proto=1)", "Penceredeki ICMP paket oranı (ICMP flood / ping sweep)."),
    "IGMP": ("Protokol", "Kolay (IP proto=2)", "Penceredeki IGMP multicast kontrol paketi oranı."),
    "IPv": ("Kopya/Kolineer", "Gereksiz (1 - ARP)", "IPv4 çerçeve oranı; ARP ile |r|=1.0 tam doğrusal bağımlı (kopya)."),
    "LLC": ("Kopya/Kolineer", "Gereksiz (1 - ARP)", "LLC katmanı oranı; ARP ve IPv ile |r|=1.0 tam doğrusal bağımlı (kopya)."),
    "Min": ("Paket Boyutu", "Kolay (min paket boyutu)", "Penceredeki en küçük paket boyutu (bayt)."),
    "Max": ("Paket Boyutu", "Kolay (max paket boyutu)", "Penceredeki en büyük paket boyutu (bayt)."),
    "AVG": ("Paket Boyutu", "Kolay (ortalama boyut)", "Penceredeki ortalama paket boyutu (bayt)."),
    "Std": ("Paket Boyutu", "Kolay (boyut std)", "Penceredeki paket boyutlarının standart sapması (flood trafiğinde ~0'a iner)."),
    "Tot size": ("Kopya/Kolineer", "Gereksiz (AVG ile ayni)", "AVG sütunu ile birebir aynı değerleri taşır (tam kopya)."),
    "IAT": ("Akis/Zaman", "Kolay (paketler arasi sure)", "Paketler arası ortalama varış süresi (Inter-Arrival Time); en güçlü davranışsal sinyal."),
    "Variance": ("Paket Boyutu", "Kısmen Kopya (Std^2)", "Paket boyutu varyansı (Std'nin karesi); canlı trafikte ikisinden biri yeterlidir."),
    "Number": ("Pencere Artefakti", "UYGUN DEGIL (Artefakt)", "Penceredeki paket sayısı; veri setinde 10 vs 100 pencere kuralını sızdırır."),
    "Tot sum": ("Pencere Artefakti", "UYGUN DEGIL (Artefakt)", "Penceredeki toplam bayt (Number * AVG); pencere boyutuna bağımlı."),
    "ack_count": ("Pencere Artefakti", "UYGUN DEGIL (Artefakt)", "Penceredeki toplam ACK sayısı; pencere boyutuna (10 vs 100) bağımlı."),
    "syn_count": ("Pencere Artefakti", "UYGUN DEGIL (Artefakt)", "Penceredeki toplam SYN sayısı; pencere boyutuna (10 vs 100) bağımlı."),
    "fin_count": ("Pencere Artefakti", "UYGUN DEGIL (Artefakt)", "Penceredeki toplam FIN sayısı; pencere boyutuna (10 vs 100) bağımlı."),
    "rst_count": ("Pencere Artefakti", "UYGUN DEGIL (Artefakt)", "Penceredeki toplam RST sayısı; pencere boyutuna (10 vs 100) bağımlı."),
}


def compute_full_binary_report(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    m = compute_binary_metrics(y_true, y_pred, y_prob)
    tn, fn = m["tn"], m["fn"]
    m["benign_precision"] = round(float(tn / (tn + fn)), 5) if (tn + fn) > 0 else 0.0
    return m


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thr, best_f1 = 0.50, -1.0
    for thr in np.linspace(0.05, 0.95, 37):
        s = f1_score(y_true, (y_prob >= thr).astype(np.int8), average="macro", zero_division=0)
        if s > best_f1:
            best_f1 = float(s)
            best_thr = float(thr)
    return round(best_thr, 4)


def stratified_subsample_idx(y: np.ndarray, max_n: int, random_state: int = 42) -> np.ndarray:
    if len(y) <= max_n:
        return np.arange(len(y))
    rng = np.random.default_rng(random_state)
    idx_list = []
    classes, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(classes, counts):
        c_idx = np.where(y == c)[0]
        take = max(1, int(round(max_n * (cnt / len(y)))))
        take = min(take, len(c_idx))
        idx_list.append(rng.choice(c_idx, size=take, replace=False))
    return np.concatenate(idx_list)


def get_comparison_models(class_weight) -> list[tuple[str, bool, object]]:
    return [
        (
            "Logistic Regression",
            True,
            LogisticRegression(max_iter=500, class_weight=class_weight, random_state=RANDOM_STATE),
        ),
        (
            "Decision Tree",
            False,
            DecisionTreeClassifier(class_weight=class_weight, random_state=RANDOM_STATE),
        ),
        (
            "Random Forest",
            False,
            RandomForestClassifier(
                n_estimators=100, class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1
            ),
        ),
        (
            "LightGBM",
            False,
            LGBMClassifier(
                n_estimators=100, class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            ),
        ),
    ]


def write_markdown_notes(
    out_md: Path,
    audit_df: pd.DataFrame,
    pruned_cols: list[str],
    dead_cols: list[str],
    pruning_df: pd.DataFrame,
    bin_comp_df: pd.DataFrame,
    mc8_comp_df: pd.DataFrame,
    train_rows: int,
    val_rows: int,
):
    lines = []
    lines.append("# CICIoT2023 Özellik Denetimi, Eleme Kararları ve Model Karşılaştırma Notları\n")
    lines.append(f"- **Eğitim Seti (File-Disjoint):** `Merged01–04` ({train_rows:,} satır)")
    lines.append(f"- **Doğrulama Seti (File-Disjoint):** `Merged07` ({val_rows:,} satır)")
    lines.append("- **Kilitli Test Seti:** `Merged09–10` (Bu aşamada hiç açılmamıştır)\n")

    lines.append("## 1. Çıkarılan Özellikler ve Gerekçeleri\n")
    lines.append("### 1.1. Pencere-Boyutu Artefaktları (6 Özellik — Kesin Çıkarıldı)")
    lines.append("CICIoT2023'te normal/düşük hacimli sınıflar 10 paketlik, DDoS/DoS/Mirai sınıfları 100 paketlik pencerelerle üretilmiştir. Canlı trafikte sabit pencere kullanılacağı için bu sütunlar ezber/sızıntı kaynağıdır:")
    for c in WINDOW_DEPENDENT_COLS:
        lines.append(f"- **`{c}`**: {FEATURE_DICTIONARY[c][2]}")

    lines.append("\n### 1.2. Matematiksel Kopya / Kolineer Sütunlar (3 Özellik — Kesin Çıkarıldı)")
    for c in EXACT_DUPLICATE_COLS:
        lines.append(f"- **`{c}`**: {FEATURE_DICTIONARY[c][2]}")

    lines.append(f"\n### 1.3. Sıfır / İhmal Edilebilir Katkılı Özellikler ({len(dead_cols)} Özellik — Deneysel Eleme)")
    if dead_cols:
        for c in dead_cols:
            row = audit_df[audit_df["feature"] == c].iloc[0]
            lines.append(
                f"- **`{c}`**: Sıfır-dışı oran=%{row['nonzero_pct']:.3f}, "
                f"Bin Gain={row['bin_gain_norm']:.5f}, MC8 Gain={row['mc8_gain_norm']:.5f}, "
                f"Val PermImp(Bin/MC8)={row['bin_perm_imp']:.5f}/{row['mc8_perm_imp']:.5f} — {FEATURE_DICTIONARY[c][2]}"
            )
    else:
        lines.append("- Tüm kalan 30 özellik pozitif katkı göstermiştir.")

    lines.append(f"\n## 2. Kalan Özellikler ve Ne İşe Yaradıkları ({len(pruned_cols)} Özellik)\n")
    lines.append("| Özellik | Kategori | Sıfır-Dışı (%) | Bin Gain | MC8 Gain | Bin PermImp | MC8 PermImp | Canlı Trafik | Açıklama |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |")
    kept_df = audit_df[audit_df["feature"].isin(pruned_cols)].sort_values("combined_perm_score", ascending=False)
    for _, r in kept_df.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['category']} | %{r['nonzero_pct']:.2f} | "
            f"{r['bin_gain_norm']:.4f} | {r['mc8_gain_norm']:.4f} | "
            f"{r['bin_perm_imp']:.5f} | {r['mc8_perm_imp']:.5f} | "
            f"{r['live_traffic_note']} | {r['description_tr']} |"
        )

    lines.append("\n## 3. Özellik Eleme ve Canlı Trafik Ablation Kıyaslaması (LightGBM, Val)\n")
    lines.append(pruning_df.to_markdown(index=False))

    lines.append("\n## 4. Tam Metrikli Model Karşılaştırması — Binary (Val)\n")
    lines.append(bin_comp_df.to_markdown(index=False))

    lines.append("\n## 5. Tam Metrikli Model Karşılaştırması — 8-Sınıf (Val)\n")
    lines.append(mc8_comp_df.to_markdown(index=False))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 2 — 39 Özellik Denetimi, Gereksiz Özellik Eleme ve Tam Metrikli Model Karşılaştırması"
    )
    parser.add_argument("--quick", action="store_true", help="Hızlı doğrulama modu (Train %3, Val %10).")
    parser.add_argument("--train-frac", type=float, default=0.25, help="Train stratified oranı (varsayılan: 0.25).")
    args = parser.parse_args()

    t_all = time.perf_counter()
    hw = get_hardware_profile()

    assert set(TRAIN_FILES).isdisjoint(set(VAL_FILES_EXP2))
    assert set(TRAIN_FILES).isdisjoint(set(TEST_FILES_LOCKED))
    assert set(VAL_FILES_EXP2).isdisjoint(set(TEST_FILES_LOCKED))

    if args.quick:
        exp_dir = RESULTS_DIR / "exp2_feature_audit_quick"
        tr_frac, va_frac, perm_n = 0.03, 0.10, 8000
    else:
        exp_dir = RESULTS_DIR / "exp2_feature_audit"
        tr_frac, va_frac, perm_n = args.train_frac, None, 50000
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================================")
    print(f"EXP 2 — ÖZELLİK DENETİMİ, ELEME VE TAM METRİKLİ MODEL KARŞILAŞTIRMASI ({'QUICK' if args.quick else 'TAM'})")
    print("=============================================================================")
    print(f"Train: {TRAIN_FILES} (frac={tr_frac}) | Val: {VAL_FILES_EXP2} | Test: {TEST_FILES_LOCKED} (KİLİTLİ)")
    print(f"Donanım: {hw['os']} | CPU={hw['logical_cpu_count']} | RAM={hw['total_ram_gb']} GB\n")

    df_tr = load_csv_files(TRAIN_FILES, feature_cols=ALL_39_FEATURES, sample_frac=tr_frac, random_state=RANDOM_STATE)
    df_va = load_csv_files(VAL_FILES_EXP2, feature_cols=ALL_39_FEATURES, sample_frac=va_frac, random_state=RANDOM_STATE)
    print(f"Train: {len(df_tr):,} satır | Val: {len(df_va):,} satır\n")

    y_tr_bin = df_tr["y_bin"].values
    y_va_bin = df_va["y_bin"].values
    le = LabelEncoder()
    le.fit(CLASS_8_ORDER)
    y_tr_8 = le.transform(df_tr["y_cat8"].values)
    y_va_8 = le.transform(df_va["y_cat8"].values)

    # =========================================================================
    # BÖLÜM 1: 30 TEMİZ ÖZELLİK ÜZERİNDE GAIN + VAL PERMUTATION IMPORTANCE
    # =========================================================================
    print("--- [1/4] 30 TEMİZ ÖZELLİK DENETİMİ (Doluluk + Gain + Val Permutation Importance) ---")
    X_tr_30 = df_tr[CLEAN_30_FEATURES]
    X_va_30 = df_va[CLEAN_30_FEATURES]

    # Std vs Variance kontrolü
    std_sq_corr = float(df_tr["Std"].fillna(0).pow(2).corr(df_tr["Variance"].fillna(0)))
    print(f"[Kontrol] corr(Std^2, Variance) = {std_sq_corr:.6f}")

    # Doğal ağırlıkla (class_weight=None) LightGBM Binary ve 8-Sınıf
    lgb_bin_30 = build_feature_pipeline(
        LGBMClassifier(n_estimators=100, class_weight=None, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    )
    lgb_bin_30.fit(X_tr_30, y_tr_bin)

    lgb_mc8_30 = build_feature_pipeline(
        LGBMClassifier(n_estimators=100, class_weight=None, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    )
    lgb_mc8_30.fit(X_tr_30, y_tr_8)

    g_bin = lgb_bin_30.named_steps["model"].booster_.feature_importance(importance_type="gain")
    g_bin_n = g_bin / max(g_bin.sum(), 1e-12)
    g_mc8 = lgb_mc8_30.named_steps["model"].booster_.feature_importance(importance_type="gain")
    g_mc8_n = g_mc8 / max(g_mc8.sum(), 1e-12)

    # Validation üzerinde stratified alt-örneklemle Permutation Importance (ezber değil genelleme katkısı)
    sub_idx_b = stratified_subsample_idx(y_va_bin, perm_n, RANDOM_STATE)
    sub_idx_8 = stratified_subsample_idx(y_va_8, perm_n, RANDOM_STATE)

    print(f"Validation Permutation Importance hesaplanıyor (n={len(sub_idx_b):,}, 3 tekrar)...")
    perm_b = permutation_importance(
        lgb_bin_30, X_va_30.iloc[sub_idx_b], y_va_bin[sub_idx_b],
        scoring="f1_macro", n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_8 = permutation_importance(
        lgb_mc8_30, X_va_30.iloc[sub_idx_8], y_va_8[sub_idx_8],
        scoring="f1_macro", n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
    )

    perm_b_map = dict(zip(CLEAN_30_FEATURES, perm_b.importances_mean))
    perm_8_map = dict(zip(CLEAN_30_FEATURES, perm_8.importances_mean))
    gain_b_map = dict(zip(CLEAN_30_FEATURES, g_bin_n))
    gain_8_map = dict(zip(CLEAN_30_FEATURES, g_mc8_n))

    audit_rows = []
    dead_cols = []

    for col in ALL_39_FEATURES:
        cat, live_note, desc_tr = FEATURE_DICTIONARY[col]
        nz_pct = float((df_tr[col].fillna(0) != 0).mean() * 100.0)
        nunq = int(df_tr[col].nunique(dropna=True))

        if col in WINDOW_DEPENDENT_COLS:
            status = "DROP_WINDOW_ARTIFACT"
            gb, g8, pb, p8 = 0.0, 0.0, 0.0, 0.0
        elif col in EXACT_DUPLICATE_COLS:
            status = "DROP_EXACT_DUPLICATE"
            gb, g8, pb, p8 = 0.0, 0.0, 0.0, 0.0
        else:
            gb = float(gain_b_map[col])
            g8 = float(gain_8_map[col])
            pb = float(perm_b_map[col])
            p8 = float(perm_8_map[col])
            # Sıfır veya ihmal edilebilir katkı kriteri:
            # Hem Binary hem 8-Sınıf gain < %0.1 VE her iki Val permutation importance <= 0.0001
            if (gb < 0.001 and g8 < 0.001) and (pb <= 0.0001 and p8 <= 0.0001):
                status = "DROP_NEGLIGIBLE_CONTRIBUTION"
                dead_cols.append(col)
            else:
                status = "KEEP"

        audit_rows.append({
            "feature": col,
            "status": status,
            "category": cat,
            "nonzero_pct": round(nz_pct, 4),
            "train_nunique": nunq,
            "bin_gain_norm": round(gb, 6),
            "mc8_gain_norm": round(g8, 6),
            "bin_perm_imp": round(pb, 6),
            "mc8_perm_imp": round(p8, 6),
            "combined_perm_score": round(max(0.0, pb) + max(0.0, p8), 6),
            "live_traffic_note": live_note,
            "description_tr": desc_tr,
        })

    audit_df = pd.DataFrame(audit_rows)
    pruned_cols = audit_df[audit_df["status"] == "KEEP"]["feature"].tolist()
    pruned_no_var_cols = [c for c in pruned_cols if c != "Variance"]
    pruned_no_port_proto_cols = [c for c in pruned_cols if c not in PORT_DERIVED_PROTOCOL_COLS]

    print(f"\nTespit edilen sıfır/ihmal edilebilir katkılı özellikler ({len(dead_cols)} adet): {dead_cols}")
    print(f"Kalan çekirdek özellik sayısı (Pruned_Core): {len(pruned_cols)} özellik")
    print("\n--- 30 TEMİZ ÖZELLİK KATKI TABLOSU (Val Permutation Importance'a Göre Sıralı) ---")
    view_30 = audit_df[audit_df["feature"].isin(CLEAN_30_FEATURES)].sort_values("combined_perm_score", ascending=False)
    print(
        view_30[
            ["feature", "status", "nonzero_pct", "bin_gain_norm", "mc8_gain_norm", "bin_perm_imp", "mc8_perm_imp"]
        ].to_string(index=False)
    )

    # =========================================================================
    # BÖLÜM 2: ELEME DOĞRULAMA + CANLI TRAFİK PROTOKOL KONTROLÜ (LIGHTGBM)
    # =========================================================================
    print("\n--- [2/4] ÖZELLİK ELEME DOĞRULAMA (Clean_30 vs Pruned vs No-Variance vs No-PortProto) ---")
    subsets = [
        ("S1_Clean_30", CLEAN_30_FEATURES),
        (f"S2_Pruned_{len(pruned_cols)}", pruned_cols),
        (f"S3_Pruned_NoVariance_{len(pruned_no_var_cols)}", pruned_no_var_cols),
        (f"S4_PureFlow_NoPortProto_{len(pruned_no_port_proto_cols)}", pruned_no_port_proto_cols),
    ]
    pruning_rows = []
    for sname, scols in subsets:
        # Binary (class_weight=None -> doğal kalibrasyon)
        pb = build_feature_pipeline(
            LGBMClassifier(n_estimators=100, class_weight=None, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        )
        pb.fit(df_tr[scols], y_tr_bin)
        ypr_b = pb.predict_proba(df_va[scols])[:, 1]
        mb = compute_full_binary_report(y_va_bin, (ypr_b >= 0.50).astype(np.int8), ypr_b)

        # 8-Class (class_weight=None)
        p8 = build_feature_pipeline(
            LGBMClassifier(n_estimators=100, class_weight=None, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        )
        p8.fit(df_tr[scols], y_tr_8)
        m8, _ = compute_multiclass_metrics(y_va_8, p8.predict(df_va[scols]), p8.predict_proba(df_va[scols]), CLASS_8_ORDER)

        pruning_rows.append({
            "feature_set": sname,
            "n_features": len(scols),
            "bin_acc": mb["accuracy"],
            "bin_atk_prec": mb["attack_precision"],
            "bin_atk_rec": mb["attack_recall"],
            "bin_ben_prec": mb["benign_precision"],
            "bin_ben_rec": mb["benign_recall"],
            "bin_fpr": mb["fpr"],
            "bin_fnr": mb["fnr"],
            "bin_fp": mb["fp"],
            "bin_fn": mb["fn"],
            "bin_f1_macro": mb["f1_macro"],
            "bin_roc_auc": mb["roc_auc"],
            "bin_pr_auc": mb["pr_auc"],
            "mc8_acc": m8["accuracy"],
            "mc8_bal_acc": m8["balanced_accuracy"],
            "mc8_prec_macro": m8["precision_macro"],
            "mc8_rec_macro": m8["recall_macro"],
            "mc8_f1_macro": m8["f1_macro"],
            "mc8_f1_weighted": m8["f1_weighted"],
            "mc8_ovr_roc": m8["roc_auc_ovr_macro"],
        })
        print(
            f"  {sname:<28} | Bin Acc={mb['accuracy']:.5f} Prec={mb['attack_precision']:.5f} Rec={mb['attack_recall']:.5f} "
            f"FPR={mb['fpr']:.4f} (FP={mb['fp']:,}, FN={mb['fn']:,}) F1={mb['f1_macro']:.5f} | "
            f"MC8 Acc={m8['accuracy']:.5f} F1_M={m8['f1_macro']:.5f}"
        )

    pruning_df = pd.DataFrame(pruning_rows)

    # =========================================================================
    # BÖLÜM 3: TAM METRİKLİ MODEL KARŞILAŞTIRMASI — BINARY (PRUNED ÖZELLİKLER)
    # =========================================================================
    print(f"\n--- [3/4] TAM METRİKLİ BINARY MODEL KARŞILAŞTIRMASI ({len(pruned_cols)} Özellik | Train vs Val Ezber Kontrolü) ---")
    X_tr_p = df_tr[pruned_cols]
    X_va_p = df_va[pruned_cols]

    bin_comp_rows = []
    # 4 model doğal ağırlıkla (class_weight=None) + LightGBM & LR için balanced kıyası
    eval_configs = [(m, sc, est, "None") for m, sc, est in get_comparison_models(None)]
    eval_configs.append((
        "LightGBM (balanced)",
        False,
        LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "balanced",
    ))

    for name, scale_num, est, cw_label in eval_configs:
        pipe = build_feature_pipeline(est, scale_numeric=scale_num)
        t0 = time.perf_counter()
        pipe.fit(X_tr_p, y_tr_bin)
        tr_sec = time.perf_counter() - t0

        # Train skoru (ezber kontrolü için)
        tr_pred = pipe.predict(X_tr_p)
        tr_f1 = round(float(f1_score(y_tr_bin, tr_pred, average="macro", zero_division=0)), 5)

        # Val default (0.50) ve tuned eşik
        va_prob = pipe.predict_proba(X_va_p)[:, 1]
        m_def = compute_full_binary_report(y_va_bin, (va_prob >= 0.50).astype(np.int8), va_prob)
        best_thr = find_best_threshold(y_va_bin, va_prob)
        m_tun = compute_full_binary_report(y_va_bin, (va_prob >= best_thr).astype(np.int8), va_prob)

        # Hangisi daha yüksek Macro-F1 ise not al ama ikisini de kaydet (tabloda default net görünsün)
        overfit_gap = round(tr_f1 - m_def["f1_macro"], 5)

        bin_comp_rows.append({
            "model": name,
            "class_weight": cw_label,
            "train_sec": round(tr_sec, 2),
            "train_f1_macro": tr_f1,
            "val_f1_macro": m_def["f1_macro"],
            "overfit_gap_f1": overfit_gap,
            "accuracy": m_def["accuracy"],
            "attack_precision": m_def["attack_precision"],
            "attack_recall": m_def["attack_recall"],
            "benign_precision": m_def["benign_precision"],
            "benign_recall": m_def["benign_recall"],
            "fpr": m_def["fpr"],
            "fnr": m_def["fnr"],
            "fp": m_def["fp"],
            "fn": m_def["fn"],
            "roc_auc": m_def["roc_auc"],
            "pr_auc": m_def["pr_auc"],
            "tuned_thr": best_thr,
            "tuned_f1_macro": m_tun["f1_macro"],
            "tuned_fpr": m_tun["fpr"],
            "tuned_attack_recall": m_tun["attack_recall"],
        })
        print(
            f"  {name:<22} | Acc={m_def['accuracy']:.5f} | AtkPrec={m_def['attack_precision']:.5f} AtkRec={m_def['attack_recall']:.5f} | "
            f"BenPrec={m_def['benign_precision']:.4f} BenRec={m_def['benign_recall']:.4f} | "
            f"FPR={m_def['fpr']:.4f} (FP={m_def['fp']:,}, FN={m_def['fn']:,}) | "
            f"F1={m_def['f1_macro']:.5f} (Gap={overfit_gap:+.4f}) | ROC={m_def['roc_auc']:.5f}"
        )

    bin_comp_df = pd.DataFrame(bin_comp_rows).sort_values("val_f1_macro", ascending=False)

    # =========================================================================
    # BÖLÜM 4: TAM METRİKLİ MODEL KARŞILAŞTIRMASI — 8-SINIF (PRUNED ÖZELLİKLER)
    # =========================================================================
    print(f"\n--- [4/4] TAM METRİKLİ 8-SINIF MODEL KARŞILAŞTIRMASI ({len(pruned_cols)} Özellik | Train vs Val Ezber Kontrolü) ---")
    mc8_comp_rows = []
    mc8_per_cls_list = []

    eval_configs_8 = [(m, sc, est, "None") for m, sc, est in get_comparison_models(None)]
    eval_configs_8.append((
        "LightGBM (balanced)",
        False,
        LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "balanced",
    ))

    for name, scale_num, est, cw_label in eval_configs_8:
        pipe = build_feature_pipeline(est, scale_numeric=scale_num)
        t0 = time.perf_counter()
        pipe.fit(X_tr_p, y_tr_8)
        tr_sec = time.perf_counter() - t0

        tr_pred = pipe.predict(X_tr_p)
        tr_f1 = round(float(f1_score(y_tr_8, tr_pred, average="macro", zero_division=0)), 5)

        va_pred = pipe.predict(X_va_p)
        va_prob = pipe.predict_proba(X_va_p)
        m8, per_cls = compute_multiclass_metrics(y_va_8, va_pred, va_prob, CLASS_8_ORDER)

        # Benign satırlarının saldırı sanılma oranı (8-sınıf FPR eşdeğeri)
        ben_idx = CLASS_8_ORDER.index("Benign")
        ben_mask = y_va_8 == ben_idx
        ben_as_attack_rate = round(float((va_pred[ben_mask] != ben_idx).mean()), 5)
        overfit_gap = round(tr_f1 - m8["f1_macro"], 5)

        mc8_comp_rows.append({
            "model": name,
            "class_weight": cw_label,
            "train_sec": round(tr_sec, 2),
            "train_f1_macro": tr_f1,
            "val_f1_macro": m8["f1_macro"],
            "overfit_gap_f1": overfit_gap,
            "accuracy": m8["accuracy"],
            "balanced_accuracy": m8["balanced_accuracy"],
            "precision_macro": m8["precision_macro"],
            "recall_macro": m8["recall_macro"],
            "f1_weighted": m8["f1_weighted"],
            "benign_misclassified_rate": ben_as_attack_rate,
            "roc_auc_ovr_macro": m8["roc_auc_ovr_macro"],
            "pr_auc_ovr_macro": m8["pr_auc_ovr_macro"],
        })
        per_cls.insert(0, "model", name)
        mc8_per_cls_list.append(per_cls)

        print(
            f"  {name:<22} | Acc={m8['accuracy']:.5f} BalAcc={m8['balanced_accuracy']:.5f} | "
            f"PrecM={m8['precision_macro']:.5f} RecM={m8['recall_macro']:.5f} | "
            f"F1_M={m8['f1_macro']:.5f} (Gap={overfit_gap:+.4f}) W-F1={m8['f1_weighted']:.5f} | "
            f"BenMis={ben_as_attack_rate:.4f} | OvR-ROC={m8['roc_auc_ovr_macro']:.5f}"
        )

    mc8_comp_df = pd.DataFrame(mc8_comp_rows).sort_values("val_f1_macro", ascending=False)
    mc8_per_cls_df = pd.concat(mc8_per_cls_list, ignore_index=True)

    # En iyi 8-sınıf modelin kategori bazlı Precision/Recall/F1 tablosunu ekrana bas
    best_mc8_name = mc8_comp_df.iloc[0]["model"]
    print(f"\n--- EN İYİ 8-SINIF MODEL ({best_mc8_name}) KATEGORİ BAZLI DETAY ---")
    print(mc8_per_cls_df[mc8_per_cls_df["model"] == best_mc8_name].drop(columns=["model"]).to_string(index=False))

    # --- KAYITLAR ---
    audit_df.to_csv(exp_dir / "all_39_features_audit.csv", index=False)
    pruning_df.to_csv(exp_dir / "pruning_comparison.csv", index=False)
    bin_comp_df.to_csv(exp_dir / "model_comparison_binary_full_metrics.csv", index=False)
    mc8_comp_df.to_csv(exp_dir / "model_comparison_mc8_full_metrics.csv", index=False)
    mc8_per_cls_df.to_csv(exp_dir / "model_comparison_mc8_per_class.csv", index=False)

    notes_path = (exp_dir / "feature_audit_notes_quick.md") if args.quick else (DOCS_DIR / "feature_audit_and_selection_notes.md")
    write_markdown_notes(
        out_md=notes_path,
        audit_df=audit_df,
        pruned_cols=pruned_cols,
        dead_cols=dead_cols,
        pruning_df=pruning_df,
        bin_comp_df=bin_comp_df,
        mc8_comp_df=mc8_comp_df,
        train_rows=len(df_tr),
        val_rows=len(df_va),
    )

    with open(exp_dir / "exp2_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "is_quick": bool(args.quick),
                "train_rows": int(len(df_tr)),
                "val_rows": int(len(df_va)),
                "std_sq_vs_variance_corr": round(std_sq_corr, 6),
                "dropped_window_artifacts": WINDOW_DEPENDENT_COLS,
                "dropped_exact_duplicates": EXACT_DUPLICATE_COLS,
                "dropped_negligible_features": dead_cols,
                "retained_pruned_features": pruned_cols,
                "hardware": hw,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nExp 2 tamamlandı ({time.perf_counter() - t_all:.1f} sn).")
    print(f"CSV çıktıları  -> {exp_dir}")
    print(f"Rapor notları  -> {notes_path}")


if __name__ == "__main__":
    main()