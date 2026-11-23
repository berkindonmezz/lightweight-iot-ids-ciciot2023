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

CAL_FILE = "Merged07.csv"   # Yalnızca karar eşiği (threshold) kalibrasyonu
EVAL_FILE = "Merged08.csv"  # Tarafsız değerlendirme & canlı trafik stres testi

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
TARGET_FPR_BUDGETS = [0.01, 0.005, 0.002, 0.001]  # %1.0, %0.5, %0.2, %0.1


def load_files_ram_safe(
    filenames: list[str],
    feature_cols: list[str],
    benign_frac: float = 1.0,
    attack_frac: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Dosya dosya okuyarak RAM'i düşük tutar; Benign ve Saldırı için ayrı oranlar destekler."""
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
    """Merged07 Benign olasılıklarından FPR <= target_fpr sağlayan en sıkı eşiği hesaplar."""
    ben_probs = np.sort(y_prob_cal[y_true_cal == 0])
    n_ben = len(ben_probs)
    allowed_fp = int(np.floor(target_fpr * n_ben))
    if allowed_fp <= 0:
        return float(min(1.0, ben_probs[-1] + 1e-6))
    if allowed_fp >= n_ben:
        return 0.0
    idx = n_ben - allowed_fp - 1
    return float(min(1.0, max(0.0, np.nextafter(ben_probs[idx], np.float64(1.0)))))


def simulate_prior_scenarios(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float, n_trials: int = 20
) -> list[dict]:
    """Merged08 üzerinde %100 Normal, %99 Normal/%1 Saldırı ve %95 Normal/%5 Saldırı simülasyonu."""
    preds = (y_prob >= thr).astype(np.int8)
    ben_idx = np.where(y_true == 0)[0]
    atk_idx = np.where(y_true == 1)[0]
    n_ben = len(ben_idx)
    fp_ben = int((preds[ben_idx] == 1).sum())
    fpr_real = fp_ben / n_ben if n_ben > 0 else 0.0

    scenarios = [{
        "scenario": "100%_Normal_0%_Attack",
        "n_benign": n_ben,
        "n_attack": 0,
        "realized_fpr": round(fpr_real, 5),
        "fp_per_10k_normal": round(fpr_real * 10000.0, 1),
        "realized_precision": np.nan,
        "realized_recall": np.nan,
        "realized_f1_attack": np.nan,
        "avg_fp": float(fp_ben),
        "avg_tp": 0.0,
        "avg_fn": 0.0,
    }]

    rng = np.random.default_rng(RANDOM_STATE)
    for label, atk_ratio in [("99%_Normal_1%_Attack", 0.01), ("95%_Normal_5%_Attack", 0.05)]:
        n_atk_need = min(len(atk_idx), max(1, int(round(n_ben * atk_ratio / (1.0 - atk_ratio)))))

        precs, recs, f1s, tps, fns = [], [], [], [], []
        for _ in range(n_trials):
            chosen_atk = rng.choice(atk_idx, size=n_atk_need, replace=False)
            tp = int((preds[chosen_atk] == 1).sum())
            fn = n_atk_need - tp
            prec = tp / (tp + fp_ben) if (tp + fp_ben) > 0 else 0.0
            rec = tp / n_atk_need if n_atk_need > 0 else 0.0
            f1_a = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1_a)
            tps.append(tp)
            fns.append(fn)

        scenarios.append({
            "scenario": label,
            "n_benign": n_ben,
            "n_attack": n_atk_need,
            "realized_fpr": round(fpr_real, 5),
            "fp_per_10k_normal": round(fpr_real * 10000.0, 1),
            "realized_precision": round(float(np.mean(precs)), 5),
            "realized_recall": round(float(np.mean(recs)), 5),
            "realized_f1_attack": round(float(np.mean(f1s)), 5),
            "avg_fp": float(fp_ben),
            "avg_tp": round(float(np.mean(tps)), 1),
            "avg_fn": round(float(np.mean(fns)), 1),
        })
    return scenarios


def family_recall_breakdown(df_eval: pd.DataFrame, y_prob: np.ndarray, thr: float) -> pd.DataFrame:
    preds = (y_prob >= thr).astype(np.int8)
    rows = []
    for cat in CLASS_8_ORDER:
        mask = df_eval["y_cat8"].values == cat
        tot = int(mask.sum())
        if cat == "Benign":
            fp = int((preds[mask] == 1).sum())
            rows.append({
                "category": cat,
                "support": tot,
                "correct_or_detected": tot - fp,
                "fp_or_missed": fp,
                "rate_pct": round((tot - fp) / tot * 100.0, 2) if tot > 0 else 0.0,
                "metric_note": "Benign Specificity (1 - FPR)",
            })
        else:
            tp = int((preds[mask] == 1).sum())
            rows.append({
                "category": cat,
                "support": tot,
                "correct_or_detected": tp,
                "fp_or_missed": tot - tp,
                "rate_pct": round(tp / tot * 100.0, 2) if tot > 0 else 0.0,
                "metric_note": "Attack Recall",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Exp 3 — Binary LightGBM Eşik Optimizasyonu ve Canlı Normal Trafik Stres Testi"
    )
    parser.add_argument("--quick", action="store_true", help="Hızlı doğrulama modu (~15-20 sn).")
    args = parser.parse_args()

    t_all = time.perf_counter()
    hw = get_hardware_profile()

    assert CAL_FILE not in TRAIN_FILES and EVAL_FILE not in TRAIN_FILES
    assert CAL_FILE not in TEST_FILES_LOCKED and EVAL_FILE not in TEST_FILES_LOCKED

    if args.quick:
        exp_dir = RESULTS_DIR / "exp3_binary_operating_point_quick"
        tr_files = TRAIN_FILES[:2]
        t1_ben, t1_atk = 0.05, 0.03
        t2_ben, t2_atk = 0.20, 0.03
        ev_ben, ev_atk = 0.15, 0.15
    else:
        exp_dir = RESULTS_DIR / "exp3_binary_operating_point"
        tr_files = TRAIN_FILES
        t1_ben, t1_atk = 0.25, 0.25
        t2_ben, t2_atk = 1.00, 0.25  # Merged01-04 içindeki %100 Benign + %25 Saldırı
        ev_ben, ev_atk = 1.00, 1.00  # Tam Merged07 ve tam Merged08

    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================================")
    print(f"EXP 3 — BINARY LIGHTGBM EŞİK & CANLI NORMAL TRAFİK STRES TESTİ ({'QUICK' if args.quick else 'TAM'})")
    print("=============================================================================")
    print(f"Özellik sayısı  : {len(FEATURES_24)} temiz özellik")
    print(f"Train           : {tr_files}")
    print(f"Eşik Kalibrasyon: {CAL_FILE} | Değerlendirme: {EVAL_FILE}")
    print(f"Kilitli Test    : {TEST_FILES_LOCKED} (DOKUNULMUYOR)")
    print(f"Donanım         : {hw['os']} | CPU={hw['logical_cpu_count']} | RAM={hw['total_ram_gb']} GB\n")

    print("Veriler yükleniyor...")
    df_t2 = load_files_ram_safe(tr_files, FEATURES_24, benign_frac=t2_ben, attack_frac=t2_atk, random_state=RANDOM_STATE)
    ben_keep_ratio = t1_ben / t2_ben
    df_t1 = pd.concat(
        [
            df_t2[df_t2["y_bin"] == 0].sample(frac=ben_keep_ratio, random_state=RANDOM_STATE),
            df_t2[df_t2["y_bin"] == 1],
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    df_cal = load_files_ram_safe([CAL_FILE], FEATURES_24, benign_frac=ev_ben, attack_frac=ev_atk, random_state=RANDOM_STATE)
    df_eval = load_files_ram_safe([EVAL_FILE], FEATURES_24, benign_frac=ev_ben, attack_frac=ev_atk, random_state=RANDOM_STATE)

    print(f"T1 (25% Benign) : {len(df_t1):,} satır (Benign={(df_t1['y_bin']==0).sum():,})")
    print(f"T2 (100% Benign): {len(df_t2):,} satır (Benign={(df_t2['y_bin']==0).sum():,})")
    print(f"Cal  ({CAL_FILE}): {len(df_cal):,} satır (Benign={(df_cal['y_bin']==0).sum():,})")
    print(f"Eval ({EVAL_FILE}): {len(df_eval):,} satır (Benign={(df_eval['y_bin']==0).sum():,})\n")

    X_cal, y_cal = df_cal[FEATURES_24], df_cal["y_bin"].values
    X_ev, y_ev = df_eval[FEATURES_24], df_eval["y_bin"].values

    # =========================================================================
    # 1) 3 LIGHTGBM KONFİGÜRASYONU VE SIKI FPR BÜTÇELERİ
    # =========================================================================
    print("--- [1/3] MODEL EĞİTİMİ VE SIKI FPR EŞİKLERİ (Kalibrasyon: Merged07 -> Test: Merged08) ---")
    configs = [
        ("LGBM_T1_25pctBenign_None", df_t1, None),
        ("LGBM_T2_100pctBenign_None", df_t2, None),
        ("LGBM_T2_100pctBenign_Balanced", df_t2, "balanced"),
    ]

    op_rows = []
    model_cal_probs = {}
    model_eval_probs = {}

    for mname, df_tr_cur, cw in configs:
        pipe = build_feature_pipeline(
            LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=63,
                class_weight=cw,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            scale_numeric=False,
        )
        t0 = time.perf_counter()
        pipe.fit(df_tr_cur[FEATURES_24], df_tr_cur["y_bin"].values)
        tr_sec = time.perf_counter() - t0

        prob_cal = pipe.predict_proba(X_cal)[:, 1]
        prob_ev = pipe.predict_proba(X_ev)[:, 1]
        model_cal_probs[mname] = prob_cal
        model_eval_probs[mname] = prob_ev

        m_base = compute_binary_metrics(y_ev, (prob_ev >= 0.50).astype(np.int8), prob_ev)
        print(f"\n>>> {mname} (Train={tr_sec:.1f}sn | ROC-AUC={m_base['roc_auc']:.5f} | PR-AUC={m_base['pr_auc']:.5f}) <<<")

        budget_list = [("Default_0.50", 0.50)] + [
            (f"Target_FPR_{b*100:.1f}%", threshold_for_target_fpr(y_cal, prob_cal, b))
            for b in TARGET_FPR_BUDGETS
        ]

        for bname, thr in budget_list:
            pred_ev = (prob_ev >= thr).astype(np.int8)
            m_ev = compute_binary_metrics(y_ev, pred_ev, prob_ev)

            op_rows.append({
                "model": mname,
                "operating_point": bname,
                "threshold_from_m07": round(thr, 6),
                "accuracy": m_ev["accuracy"],
                "f1_macro": m_ev["f1_macro"],
                "attack_precision": m_ev["attack_precision"],
                "attack_recall": m_ev["attack_recall"],
                "benign_recall": m_ev["benign_recall"],
                "fpr": m_ev["fpr"],
                "fp_per_10k_normal": round(m_ev["fpr"] * 10000.0, 1),
                "fp": m_ev["fp"],
                "fn": m_ev["fn"],
                "roc_auc": m_ev["roc_auc"],
                "pr_auc": m_ev["pr_auc"],
            })

            print(
                f"  {bname:<16} (t={thr:.5f}) | Acc={m_ev['accuracy']:.5f} | "
                f"AtkPrec={m_ev['attack_precision']:.5f} AtkRec={m_ev['attack_recall']:.5f} | "
                f"FPR={m_ev['fpr']*100:.3f}% (10k'da {m_ev['fpr']*10000:.1f} FP | FP={m_ev['fp']:<4}, FN={m_ev['fn']:<6}) | "
                f"F1_M={m_ev['f1_macro']:.5f}"
            )

    op_df = pd.DataFrame(op_rows)

    # Target_FPR_0.5% noktasında en yüksek Attack Recall veren modeli seç
    sub_05 = op_df[op_df["operating_point"] == "Target_FPR_0.5%"].sort_values("attack_recall", ascending=False)
    best_model_name = sub_05.iloc[0]["model"]
    best_prob_cal = model_cal_probs[best_model_name]
    best_prob_ev = model_eval_probs[best_model_name]

    # =========================================================================
    # 2) CANLI NORMAL TRAFİK STRES TESTİ (%100 / %99 / %95 NORMAL)
    # =========================================================================
    print(f"\n--- [2/3] CANLI NORMAL TRAFİK STRES TESTİ ({best_model_name} | Merged08) ---")
    stress_rows = []
    for b in TARGET_FPR_BUDGETS:
        bname = f"Target_FPR_{b*100:.1f}%"
        thr = threshold_for_target_fpr(y_cal, best_prob_cal, b)
        for sc in simulate_prior_scenarios(y_ev, best_prob_ev, thr, n_trials=20):
            sc["model"] = best_model_name
            sc["operating_point"] = bname
            sc["threshold"] = round(thr, 6)
            stress_rows.append(sc)

    stress_df = pd.DataFrame(stress_rows)
    print(
        stress_df[
            [
                "operating_point",
                "scenario",
                "realized_fpr",
                "fp_per_10k_normal",
                "realized_precision",
                "realized_recall",
                "realized_f1_attack",
                "avg_fp",
                "avg_tp",
                "avg_fn",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # 3) 7 SALDIRI AİLESİ YAKALAMA ANALİZİ
    # =========================================================================
    print(f"\n--- [3/3] 7 SALDIRI AİLESİ HATA ANALİZİ ({best_model_name} | Merged08) ---")
    fam_frames = []
    for b in TARGET_FPR_BUDGETS:
        bname = f"Target_FPR_{b*100:.1f}%"
        thr = threshold_for_target_fpr(y_cal, best_prob_cal, b)
        fdf = family_recall_breakdown(df_eval, best_prob_ev, thr)
        fdf.insert(0, "operating_point", bname)
        fam_frames.append(fdf)
        if b in (0.005, 0.001):
            print(f"\n  [{bname} | t={thr:.5f}]")
            print(fdf.drop(columns=["operating_point"]).to_string(index=False))

    fam_df = pd.concat(fam_frames, ignore_index=True)

    # --- KAYITLAR ---
    op_df.to_csv(exp_dir / "operating_points_comparison.csv", index=False)
    stress_df.to_csv(exp_dir / "live_prior_stress_test.csv", index=False)
    fam_df.to_csv(exp_dir / "attack_family_recall_breakdown.csv", index=False)

    with open(exp_dir / "exp3_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "is_quick": bool(args.quick),
                "features_24": FEATURES_24,
                "train_files": tr_files,
                "cal_file": CAL_FILE,
                "eval_file": EVAL_FILE,
                "locked_test_files": TEST_FILES_LOCKED,
                "best_model_at_fpr_0_5pct": best_model_name,
                "hardware": hw,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not args.quick:
        md_lines = [
            "# Exp 3 — Binary LightGBM Eşik Optimizasyonu ve Canlı Normal Trafik Stres Testi\n",
            f"- **Özellik Seti (24 Özellik):** `{', '.join(FEATURES_24)}`",
            f"- **Bölme:** Train=`Merged01–04`, Eşik Kalibrasyonu=`{CAL_FILE}`, Değerlendirme=`{EVAL_FILE}`, Kilitli Test=`{TEST_FILES_LOCKED}`\n",
            "## 1. FPR Bütçelerine Göre Çalışma Noktaları (`Merged08`)\n",
            op_df.to_markdown(index=False),
            f"\n## 2. Canlı Normal Trafik Stres Testi ({best_model_name})\n",
            stress_df.to_markdown(index=False),
            f"\n## 3. Saldırı Ailesi Bazlı Yakalama Oranları ({best_model_name})\n",
            fam_df.to_markdown(index=False),
        ]
        notes_path = DOCS_DIR / "exp3_operating_point_notes.md"
        notes_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"\nRapor notları -> {notes_path}")

    print(f"Exp 3 tamamlandı ({time.perf_counter() - t_all:.1f} sn). Çıktılar -> {exp_dir}")


if __name__ == "__main__":
    main()