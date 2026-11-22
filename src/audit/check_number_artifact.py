import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "raw_ciciot2023"

WINDOW_DEPENDENT_COLS = [
    "Number",
    "Tot sum",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
]


def map_to_8_classes(label: str) -> str:
    s = str(label).strip().upper()
    if s == "BENIGN":
        return "Benign"
    if s.startswith("DDOS"):
        return "DDoS"
    if s.startswith("DOS"):
        return "DoS"
    if s.startswith("MIRAI"):
        return "Mirai"
    if s.startswith("RECON") or s == "VULNERABILITYSCAN":
        return "Recon"
    if s in ("MITM-ARPSPOOFING", "DNS_SPOOFING"):
        return "Spoofing"
    if s == "DICTIONARYBRUTEFORCE":
        return "BruteForce"
    return "Web"


def eval_subset(name: str, cols: list[str], df_tr: pd.DataFrame, df_va: pd.DataFrame, y_tr_bin, y_va_bin, y_tr_8, y_va_8, le):
    X_tr = df_tr[cols].replace([np.inf, -np.inf], np.nan)
    X_va = df_va[cols].replace([np.inf, -np.inf], np.nan)

    # Binary
    pipe_b = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
    ])
    pipe_b.fit(X_tr, y_tr_bin)
    yp_b = pipe_b.predict(X_va)
    ypr_b = pipe_b.predict_proba(X_va)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_va_bin, yp_b, labels=[0, 1]).ravel()
    fpr = fp / (tn + fp)
    rec = tp / (tp + fn)
    f1_b = f1_score(y_va_bin, yp_b, average="macro")
    roc_b = roc_auc_score(y_va_bin, ypr_b)
    pr_b = average_precision_score(y_va_bin, ypr_b)

    gains = pipe_b.named_steps["clf"].booster_.feature_importance(importance_type="gain")
    gains_norm = gains / gains.sum()
    top5 = sorted(zip(cols, gains_norm), key=lambda x: x[1], reverse=True)[:5]
    top5_str = ", ".join(f"{c}({v:.2f})" for c, v in top5)

    # 8-Class
    pipe_8 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
    ])
    pipe_8.fit(X_tr, y_tr_8)
    yp_8 = pipe_8.predict(X_va)
    f1_8_m = f1_score(y_va_8, yp_8, average="macro")
    f1_8_w = f1_score(y_va_8, yp_8, average="weighted")

    print(f"\n>>> {name} ({len(cols)} özellik) <<<")
    print(f"  [Binary ] Macro-F1={f1_b:.5f} | Recall={rec:.5f} | FPR={fpr:.5f} (FP={fp:,}, FN={fn:,}) | ROC={roc_b:.5f} | PR={pr_b:.5f}")
    print(f"  [Top-5  ] {top5_str}")
    print(f"  [8-Sınıf] Macro-F1={f1_8_m:.5f} | Weighted-F1={f1_8_w:.5f}")


def main():
    t0 = time.perf_counter()
    df_tr = pd.read_csv(DATA_DIR / "Merged01.csv")
    df_va = pd.read_csv(DATA_DIR / "Merged07.csv")

    label_col = "Label"
    all_feats = [c for c in df_tr.columns if c != label_col]

    # 1. Number sütununun 8 sınıfa göre dağılımı
    df_tr["cat8"] = df_tr[label_col].map(map_to_8_classes)
    print("=== 1. 'Number' SÜTUNUNUN 8 SINIFA GÖRE DAĞILIMI (Merged01) ===")
    num_stats = df_tr.groupby("cat8")["Number"].agg(["count", "min", "median", "mean", "max"]).round(2)
    print(num_stats.to_string())

    # Etiketler
    y_tr_bin = (df_tr[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(np.int8).values
    y_va_bin = (df_va[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(np.int8).values

    le = LabelEncoder()
    le.fit(sorted(df_tr["cat8"].unique()))
    y_tr_8 = le.transform(df_tr["cat8"].values)
    y_va_8 = le.transform(df_va[label_col].map(map_to_8_classes).values)

    # 2. İki temiz alt küme testi
    cols_no_num = [c for c in all_feats if c != "Number"]
    cols_scale_inv = [c for c in all_feats if c not in WINDOW_DEPENDENT_COLS]

    eval_subset("Sadece Number Hariç", cols_no_num, df_tr, df_va, y_tr_bin, y_va_bin, y_tr_8, y_va_8, le)
    eval_subset("Pencere-Boyutu Bağımsız (Number + Tot sum + *_count Hariç)", cols_scale_inv, df_tr, df_va, y_tr_bin, y_va_bin, y_tr_8, y_va_8, le)

    print(f"\nTamamlandı ({time.perf_counter() - t0:.1f} sn).")


if __name__ == "__main__":
    main()