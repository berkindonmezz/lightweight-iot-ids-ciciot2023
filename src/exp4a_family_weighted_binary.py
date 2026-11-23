import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import compute_binary_metrics, get_hardware_profile
from src.config import (
    CLASS_8_ORDER,
    DOCS_DIR,
    FEATURES_33,
    LABEL_COL,
    RANDOM_STATE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TEST_FILES_LOCKED,
    TRAIN_FILES,
    map_label_to_8_classes,
)
from src.preprocessing import build_feature_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

CAL_FILE = "Merged07.csv"
EVAL_FILE = "Merged08.csv"

DROPPED_FROM_33 = {
    "Tot size",
    "IPv",
    "LLC",
    "Variance",
    "cwr_flag_number",
    "SMTP",
    "Telnet",
    "IRC",
    "IGMP",
}
FEATURES_24 = [c for c in FEATURES_33 if c not in DROPPED_FROM_33]
TARGET_FPR_BUDGETS = [0.005, 0.002, 0.001]  # %0.5, %0.2, %0.1

# 3 ağırlık rejimi (yalnızca eğitim sırasında kullanılır; model boyutu/hızı değişmez)
WEIGHT_SCHEMES = {
    "W0_Uniform": {
        "Benign": 1.0,
        "DDoS": 1.0,
        "DoS": 1.0,
        "Mirai": 1.0,
        "Spoofing": 1.0,
        "Recon": 1.0,
        "BruteForce": 1.0,
        "Web": 1.0,
    },
    "W1_Mild_Boost": {
        "Benign": 1.0,
        "DDoS": 1.0,
        "DoS": 1.0,
        "Mirai": 1.0,
        "Spoofing": 4.0,
        "Recon": 5.0,
        "BruteForce": 15.0,
        "Web": 15.0,
    },
    "W2_Strong_Boost": {
        "Benign": 2.0,
        "DDoS": 1.0,
        "DoS": 1.0,
        "Mirai": 1.0,
        "Spoofing": 10.0,
        "Recon": 15.0,
        "BruteForce": 40.0,
        "Web": 40.0,
    },
}


def load_files_ram_safe(
    filenames: list[str],
    feature_cols: list[str],
    benign_frac: float = 1.0,
    attack_frac: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    usecols = list(dict.fromkeys(feature_cols + [LABEL_COL]))
    parts = []
    for i, fname in enumerate(filenames):
        fpath = RAW_DATA_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {fpath}")

        df_p = pd.read_csv(fpath, usecols=usecols, low_memory=False)
        for col in feature_cols:
            if np.issubdtype(df_p[col].dtype, np.number):
                df_p[col] = df_p[col].replace([np.inf, -np.inf], np.nan).astype(np.float32)

        raw_lbl = df_p[LABEL_COL].astype(str).str.strip()
        df_p["y_bin"] = (raw_lbl.str.upper() != "BENIGN").astype(np.int8)
        df_p["y_cat8"] = raw_lbl.map(map_label_to_8_classes)

        sub_frames = []
        df_ben = df_p[df_p["y_bin"] == 0]
        if benign_frac < 1.0:
            n_b = max(1, int(round(len(df_ben) * benign_frac)))
            df_ben = df_ben.sample(n=min(n_b, len(df_ben)), random_state=random_state + i)
        sub_frames.append(df_ben)

        df_atk = df_p[df_p["y_bin"] == 1]
        if attack_frac < 1.0:
            for j, (_, grp) in enumerate(df_atk.groupby("y_cat8", sort=False)):
                n_a = max(1, int(round(len(grp) * attack_frac)))
                sub_frames.append(grp.sample(n=min(n_a, len(grp)), random_state=random_state + i * 10 + j))
        else:
            sub_frames.append(df_atk)

        parts.append(pd.concat(sub_frames, ignore_index=True))
        del df_p, df_ben, df_atk, sub_frames
        gc.collect()

    df_all = pd.concat(parts, ignore_index=True)
    return df_all.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def threshold_for_target_fpr(y_true_cal: np.ndarray, y_prob_cal: np.ndarray, target_fpr: float) -> float:
    ben_probs = np.sort(y_prob_cal[y_true_cal == 0])
    n_ben = len(ben_probs)
    allowed_fp = int(np.floor(target_fpr * n_ben))
    if allowed_fp <= 0:
        return float(min(1.0, ben_probs[-1] + 1e-6))
    if allowed_fp >= n_ben:
        return 0.0
    idx = n_ben - allowed_fp - 1
    return float(min(1.0, max(0.0, np.nextafter(ben_probs[idx], np.float64(1.0)))))


def compute_99pct_normal_metrics(y_true: np.ndarray, preds: np.ndarray, n_trials: int = 20) -> tuple[float, float]:
    """%99 Normal / %1 Saldırı senaryosunda ortalama Precision ve Attack-F1 hesaplar."""
    ben_idx = np.where(y_true == 0)[0]
    atk_idx = np.where(y_true == 1)[0]
    fp_ben = int((preds[ben_idx] == 1).sum())
    n_atk_need = min(len(atk_idx), max(1, int(round(len(ben_idx) * 0.01 / 0.99))))

    rng = np.random.default_rng(RANDOM_STATE)
    precs, f1s = [], []
    for _ in range(n_trials):
        chosen = rng.choice(atk_idx, size=n_atk_need, replace=False)
        tp = int((preds[chosen] == 1).sum())
        prec = tp / (tp + fp_ben) if (tp + fp_ben) > 0 else 0.0
        rec = tp / n_atk_need if n_atk_need > 0 else 0.0
        f1_a = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        f1s.append(f1_a)
    return round(float(np.mean(precs)), 5), round(float(np.mean(f1s)), 5)


def main():
    parser = argparse.ArgumentParser(
        description="Exp 4a — Saldırı Ailesi Ağırlıklı Binary LightGBM Deneyi (Sıfır Ek Maliyet Kontrolü)"
    )
    parser.add_argument("--quick", action="store_true", help="Hızlı doğrulama modu (~15 sn).")
    args = parser.parse_args()

    t_all = time.perf_counter()
    hw = get_hardware_profile()

    assert CAL_FILE not in TRAIN_FILES and EVAL_FILE not in TRAIN_FILES
    assert CAL_FILE not in TEST_FILES_LOCKED and EVAL_FILE not in TEST_FILES_LOCKED

    if args.quick:
        exp_dir = RESULTS_DIR / "exp4a_family_weighted_quick"
        tr_files = TRAIN_FILES[:2]
        tr_ben, tr_atk, ev_frac = 0.20, 0.03, 0.15
    else:
        exp_dir = RESULTS_DIR / "exp4a_family_weighted"
        tr_files = TRAIN_FILES
        tr_ben, tr_atk, ev_frac = 1.00, 0.25, 1.00

    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================================")
    print(f"EXP 4A — SALDIRI AİLESİ AĞIRLIKLI BINARY LIGHTGBM ({'QUICK' if args.quick else 'TAM'})")
    print("=============================================================================")
    print(f"Özellikler: {len(FEATURES_24)} temiz özellik | Train: {tr_files} | Cal: {CAL_FILE} | Eval: {EVAL_FILE}\n")

    df_tr = load_files_ram_safe(tr_files, FEATURES_24, benign_frac=tr_ben, attack_frac=tr_atk)
    df_cal = load_files_ram_safe([CAL_FILE], FEATURES_24, benign_frac=ev_frac, attack_frac=ev_frac)
    df_eval = load_files_ram_safe([EVAL_FILE], FEATURES_24, benign_frac=ev_frac, attack_frac=ev_frac)

    X_tr, y_tr = df_tr[FEATURES_24], df_tr["y_bin"].values
    X_cal, y_cal = df_cal[FEATURES_24], df_cal["y_bin"].values
    X_ev, y_ev = df_eval[FEATURES_24], df_eval["y_bin"].values
    cat_ev = df_eval["y_cat8"].values

    print(f"Train: {len(df_tr):,} (Benign={(y_tr==0).sum():,}) | Cal: {len(df_cal):,} | Eval: {len(df_eval):,}\n")

    summary_rows = []
    family_rows = []

    for scheme_name, w_map in WEIGHT_SCHEMES.items():
        sw = df_tr["y_cat8"].map(w_map).astype(np.float32).values

        pipe = build_feature_pipeline(
            LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=63,
                class_weight=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            scale_numeric=False,
        )
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr, model__sample_weight=sw)
        tr_sec = time.perf_counter() - t0

        prob_cal = pipe.predict_proba(X_cal)[:, 1]
        prob_ev = pipe.predict_proba(X_ev)[:, 1]

        print(f">>> {scheme_name} (Train={tr_sec:.1f} sn) <<<")
        for b in TARGET_FPR_BUDGETS:
            bname = f"Target_FPR_{b*100:.1f}%"
            thr = threshold_for_target_fpr(y_cal, prob_cal, b)
            pred_ev = (prob_ev >= thr).astype(np.int8)
            m = compute_binary_metrics(y_ev, pred_ev, prob_ev)
            live99_prec, live99_f1 = compute_99pct_normal_metrics(y_ev, pred_ev)

            # Saldırı ailesi recall'ları (özellikle zayıf 4 sınıf)
            fam_rec = {}
            for cat in CLASS_8_ORDER:
                cmask = cat_ev == cat
                tot = int(cmask.sum())
                if cat == "Benign":
                    val_pct = round(float((pred_ev[cmask] == 0).mean() * 100.0), 2) if tot else 0.0
                else:
                    val_pct = round(float((pred_ev[cmask] == 1).mean() * 100.0), 2) if tot else 0.0
                fam_rec[cat] = val_pct
                family_rows.append({
                    "scheme": scheme_name,
                    "operating_point": bname,
                    "category": cat,
                    "support": tot,
                    "rate_pct": val_pct,
                })

            summary_rows.append({
                "scheme": scheme_name,
                "operating_point": bname,
                "threshold_m07": round(thr, 6),
                "accuracy": m["accuracy"],
                "attack_recall": m["attack_recall"],
                "fpr": m["fpr"],
                "fp": m["fp"],
                "fn": m["fn"],
                "roc_auc": m["roc_auc"],
                "live99_precision": live99_prec,
                "live99_f1_attack": live99_f1,
                "rec_DDoS_%": fam_rec["DDoS"],
                "rec_DoS_%": fam_rec["DoS"],
                "rec_Mirai_%": fam_rec["Mirai"],
                "rec_Spoofing_%": fam_rec["Spoofing"],
                "rec_Recon_%": fam_rec["Recon"],
                "rec_BruteForce_%": fam_rec["BruteForce"],
                "rec_Web_%": fam_rec["Web"],
            })

            print(
                f"  {bname:<15} | FPR={m['fpr']*100:.3f}% (FP={m['fp']:<3}) | AtkRec={m['attack_recall']:.5f} | "
                f"Live99(P={live99_prec:.3f}, F1={live99_f1:.3f}) | "
                f"Spoof={fam_rec['Spoofing']:5.1f}% Recon={fam_rec['Recon']:5.1f}% "
                f"Brute={fam_rec['BruteForce']:5.1f}% Web={fam_rec['Web']:5.1f}%"
            )
        print()

    sum_df = pd.DataFrame(summary_rows)
    fam_df = pd.DataFrame(family_rows)

    print("=== YAN YANA ÖZET: Target_FPR_0.5% VE Target_FPR_0.1% ===")
    cols_view = [
        "scheme", "operating_point", "fpr", "fp", "attack_recall",
        "live99_precision", "rec_Spoofing_%", "rec_Recon_%", "rec_BruteForce_%", "rec_Web_%",
    ]
    print(sum_df[sum_df["operating_point"].isin(["Target_FPR_0.5%", "Target_FPR_0.1%"])][cols_view].to_string(index=False))

    sum_df.to_csv(exp_dir / "family_weighted_summary.csv", index=False)
    fam_df.to_csv(exp_dir / "family_weighted_per_category.csv", index=False)
    with open(exp_dir / "exp4a_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"is_quick": bool(args.quick), "weight_schemes": WEIGHT_SCHEMES, "hardware": hw}, f, indent=2)

    if not args.quick:
        md_lines = [
            "# Exp 4a — Saldırı Ailesi Ağırlıklı Binary LightGBM Deneyi\n",
            "Amaç: Model boyutu ve gecikmesini hiç artırmadan yalnızca eğitim `sample_weight` değerleriyle düşük hacimli saldırıların yakalanma oranını test etmek.\n",
            "## Karşılaştırma Tablosu (`Merged08`)\n",
            sum_df.to_markdown(index=False),
        ]
        notes_path = DOCS_DIR / "exp4a_family_weighted_notes.md"
        notes_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"\nRapor notları -> {notes_path}")

    print(f"Exp 4a tamamlandı ({time.perf_counter() - t_all:.1f} sn). Çıktılar -> {exp_dir}")


if __name__ == "__main__":
    main()