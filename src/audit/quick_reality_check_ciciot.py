import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "raw_ciciot2023"


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
    return "Web"  # SQLInjection, XSS, CommandInjection, BrowserHijacking, Backdoor, Uploading


def clean_inf(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    return X


def main():
    t0 = time.perf_counter()
    print("Yükleniyor: Merged01 (Train) ve Merged07 (Val)...")
    df_tr = pd.read_csv(DATA_DIR / "Merged01.csv")
    df_va = pd.read_csv(DATA_DIR / "Merged07.csv")

    label_col = "Label"
    feat_cols = [c for c in df_tr.columns if c != label_col]

    X_tr = clean_inf(df_tr, feat_cols)
    X_va = clean_inf(df_va, feat_cols)

    # --- 1) BINARY GÖREV (BENIGN=0 vs ATTACK=1) ---
    y_tr_bin = (df_tr[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(np.int8).values
    y_va_bin = (df_va[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(np.int8).values

    pipe_bin = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
    ])
    t_b = time.perf_counter()
    pipe_bin.fit(X_tr, y_tr_bin)
    y_pred_b = pipe_bin.predict(X_va)
    y_prob_b = pipe_bin.predict_proba(X_va)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_va_bin, y_pred_b, labels=[0, 1]).ravel()
    fpr = fp / (tn + fp)
    rec = tp / (tp + fn)
    f1_m = f1_score(y_va_bin, y_pred_b, average="macro")
    pr_auc = average_precision_score(y_va_bin, y_prob_b)
    roc_auc = roc_auc_score(y_va_bin, y_prob_b)

    print(f"\n=== 1. BINARY SONUÇ (Merged01 -> Merged07, {time.perf_counter() - t_b:.1f} sn) ===")
    print(f"Macro-F1 : {f1_m:.5f} | Attack Recall: {rec:.5f} | FPR: {fpr:.5f} (FP={fp:,}, FN={fn:,})")
    print(f"ROC-AUC  : {roc_auc:.5f} | PR-AUC: {pr_auc:.5f}")

    # En önemli 10 özellik (Gain)
    gains = pipe_bin.named_steps["model"].booster_.feature_importance(importance_type="gain")
    gains_norm = gains / gains.sum()
    imp_df = pd.DataFrame({"Ozellik": feat_cols, "Gain_Orani": np.round(gains_norm, 4)}).sort_values("Gain_Orani", ascending=False)
    print("\n--- Binary Model En Önemli 10 Özellik (Gain) ---")
    print(imp_df.head(10).to_string(index=False))

    # --- 2) 8-SINIF GÖREV (Benign + 7 Ana Kategori) ---
    y_tr_8_str = df_tr[label_col].map(map_to_8_classes).values
    y_va_8_str = df_va[label_col].map(map_to_8_classes).values

    le = LabelEncoder()
    le.fit(sorted(np.unique(y_tr_8_str)))
    y_tr_8 = le.transform(y_tr_8_str)
    y_va_8 = le.transform(y_va_8_str)

    pipe_8 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
    ])
    t_8 = time.perf_counter()
    pipe_8.fit(X_tr, y_tr_8)
    y_pred_8 = pipe_8.predict(X_va)

    f1_8_m = f1_score(y_va_8, y_pred_8, average="macro")
    f1_8_w = f1_score(y_va_8, y_pred_8, average="weighted")
    print(f"\n=== 2. 8-SINIF SONUÇ (Merged01 -> Merged07, {time.perf_counter() - t_8:.1f} sn) ===")
    print(f"Macro-F1 : {f1_8_m:.5f} | Weighted-F1: {f1_8_w:.5f}")
    print("\n--- 8-Sınıf Kategori Bazlı F1 ---")
    f1_per_cls = f1_score(y_va_8, y_pred_8, average=None, labels=range(len(le.classes_)))
    for cname, score in zip(le.classes_, f1_per_cls):
        sup = int((y_va_8_str == cname).sum())
        print(f"  {cname:<12}: F1={score:.4f} (support={sup:,})")

    print(f"\nToplam süre: {time.perf_counter() - t0:.1f} sn")


if __name__ == "__main__":
    main()