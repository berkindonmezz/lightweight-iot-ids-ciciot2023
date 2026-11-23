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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    get_hardware_profile,
    measure_edge_cost,
)
from config import (
    CLASS_8_ORDER,
    FEATURES_32_NO_HTTPS,
    FEATURES_33,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_FILES_LOCKED,
    TRAIN_FILES,
    WINDOW_DEPENDENT_COLS,
)
from preprocessing import build_feature_pipeline, load_csv_files

warnings.filterwarnings("ignore", category=UserWarning)

# Validation olarak Merged07 tam dosya (702k satır); Merged09-10 kesinlikle kilitli
EXP1_VAL_FILES = ["Merged07.csv"]
ALL_39_FEATURES = FEATURES_33 + WINDOW_DEPENDENT_COLS


def get_model_specs() -> list[tuple[str, str, bool, object]]:
    return [
        (
            "Logistic Regression",
            "logreg",
            True,
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Decision Tree",
            "dt",
            False,
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            "rf",
            False,
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "LightGBM",
            "lgbm",
            False,
            LGBMClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]


def find_best_binary_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Validation olasılıkları üzerinde Binary Macro-F1'i maksimize eden eşiği bulur."""
    best_thr = 0.50
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 37):
        preds = (y_prob >= thr).astype(np.int8)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return round(best_thr, 4), round(best_f1, 5)


def check_collinearity(df: pd.DataFrame, cols: list[str]) -> dict:
    """AVG vs Tot size ve yüksek korelasyonlu (|r| >= 0.999) sütun çiftlerini raporlar."""
    avg_tot_exact = bool(np.allclose(df["AVG"].fillna(0).values, df["Tot size"].fillna(0).values, atol=1e-6))
    corr = df[cols].corr().abs()
    high_pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and val >= 0.999:
                high_pairs.append({"col_a": cols[i], "col_b": cols[j], "abs_corr": round(float(val), 6)})
    return {
        "avg_equals_tot_size": avg_tot_exact,
        "near_duplicate_pairs_ge_0999": high_pairs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CICIoT2023 Exp 1 — Baseline (Binary + 8-Class), Threshold Tuning & Ablation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Hızlı doğrulama modu (Train %3, Val %10 örneklem — results/exp1_baselines_quick/ altına yazar).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.25,
        help="Tam modda Merged01-04 eğitim dosyalarından alınacak stratified oran (varsayılan: 0.25 ≈ 708k satır).",
    )
    args = parser.parse_args()

    t_all = time.perf_counter()
    hw = get_hardware_profile()

    # Dosya-ayrık bölme ve kilitli test doğrulaması
    assert set(TRAIN_FILES).isdisjoint(set(EXP1_VAL_FILES))
    assert set(TRAIN_FILES).isdisjoint(set(TEST_FILES_LOCKED))
    assert set(EXP1_VAL_FILES).isdisjoint(set(TEST_FILES_LOCKED))

    if args.quick:
        exp_dir = RESULTS_DIR / "exp1_baselines_quick"
        mod_dir = MODELS_DIR / "exp1_baselines_quick"
        tr_frac = 0.03
        va_frac = 0.10
        n_lat = 40
    else:
        exp_dir = RESULTS_DIR / "exp1_baselines"
        mod_dir = MODELS_DIR / "exp1_baselines"
        tr_frac = args.train_frac
        va_frac = None  # Tam Merged07 (702,212 satır)
        n_lat = 200

    exp_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================================")
    print(f"CICIOT2023 EXP 1 — BASELINE & ABLATION ({'QUICK TEST' if args.quick else 'TAM DENEY'})")
    print("=============================================================================")
    print(f"Train dosyaları : {TRAIN_FILES} (frac={tr_frac})")
    print(f"Val dosyaları   : {EXP1_VAL_FILES} (frac={va_frac if va_frac else '1.00 - TAM'})")
    print(f"Test dosyaları  : {TEST_FILES_LOCKED} (KİLİTLİ — KULLANILMIYOR)")
    print(f"Donanım         : {hw['os']} | CPU={hw['logical_cpu_count']} | RAM={hw['total_ram_gb']} GB\n")

    print("Veriler yükleniyor...")
    df_tr = load_csv_files(TRAIN_FILES, feature_cols=ALL_39_FEATURES, sample_frac=tr_frac, random_state=RANDOM_STATE)
    df_va = load_csv_files(EXP1_VAL_FILES, feature_cols=ALL_39_FEATURES, sample_frac=va_frac, random_state=RANDOM_STATE)

    print(f"Train: {len(df_tr):,} satır (Benign=%{(df_tr['y_bin']==0).mean()*100:.2f}) | "
          f"Val: {len(df_va):,} satır (Benign=%{(df_va['y_bin']==0).mean()*100:.2f})")

    # 0) Kolinearite kontrolü (33 özellik)
    collin = check_collinearity(df_tr, FEATURES_33)
    print(f"\n[Kolinearite Kontrolü] AVG == Tot size mı?: {collin['avg_equals_tot_size']}")
    if collin["near_duplicate_pairs_ge_0999"]:
        print("  |r| >= 0.999 çiftler:", collin["near_duplicate_pairs_ge_0999"])
    else:
        print("  |r| >= 0.999 çift yok.")

    X_tr_33 = df_tr[FEATURES_33]
    X_va_33 = df_va[FEATURES_33]
    y_tr_bin = df_tr["y_bin"].values
    y_va_bin = df_va["y_bin"].values

    le = LabelEncoder()
    le.fit(CLASS_8_ORDER)
    y_tr_8 = le.transform(df_tr["y_cat8"].values)
    y_va_8 = le.transform(df_va["y_cat8"].values)

    # =====================================================================
    # 1) BINARY BASELINE (33 PENCERE-BAĞIMSIZ ÖZELLİK)
    # =====================================================================
    print("\n--- [1/3] BINARY BASELINE (33 Temiz Özellik, Default vs. Val-Tuned Threshold) ---")
    bin_rows = []
    for name, tag, scale_num, est in get_model_specs():
        pipe = build_feature_pipeline(est, scale_numeric=scale_num)
        t0 = time.perf_counter()
        pipe.fit(X_tr_33, y_tr_bin)
        tr_sec = time.perf_counter() - t0

        y_prob = pipe.predict_proba(X_va_33)[:, 1]
        y_pred_def = (y_prob >= 0.50).astype(np.int8)
        m_def = compute_binary_metrics(y_va_bin, y_pred_def, y_prob)

        best_thr, _ = find_best_binary_threshold(y_va_bin, y_prob)
        y_pred_tun = (y_prob >= best_thr).astype(np.int8)
        m_tun = compute_binary_metrics(y_va_bin, y_pred_tun, y_prob)

        edge = measure_edge_cost(
            pipeline=pipe,
            X_eval=X_va_33,
            save_pipeline_path=mod_dir / f"exp1_bin_{tag}_pipe.joblib",
            save_model_only_path=mod_dir / f"exp1_bin_{tag}_model.joblib",
            train_time_sec=tr_sec,
            n_lat_samples=n_lat,
        )

        bin_rows.append({
            "model": name,
            "n_features": len(FEATURES_33),
            "default_thr": 0.50,
            "default_f1_macro": m_def["f1_macro"],
            "default_fpr": m_def["fpr"],
            "default_attack_recall": m_def["attack_recall"],
            "tuned_thr_on_val": best_thr,
            "tuned_f1_macro": m_tun["f1_macro"],
            "tuned_fpr": m_tun["fpr"],
            "tuned_attack_recall": m_tun["attack_recall"],
            "tuned_fp": m_tun["fp"],
            "tuned_fn": m_tun["fn"],
            "roc_auc": m_def["roc_auc"],
            "pr_auc": m_def["pr_auc"],
            **edge,
        })

        print(
            f"  [Bin ] {name:<20} | Def F1={m_def['f1_macro']:.4f} (FPR={m_def['fpr']:.4f}) -> "
            f"Tuned(t={best_thr:.2f}) F1={m_tun['f1_macro']:.4f} (FPR={m_tun['fpr']:.4f}, Rec={m_tun['attack_recall']:.4f}) | "
            f"Size={edge['model_only_size_kb']:.1f}KB | Lat(M/E2E P50)={edge['model_only_latency_p50_ms']:.3f}/{edge['e2e_latency_p50_ms']:.3f}ms"
        )

    # =====================================================================
    # 2) 8-SINIF BASELINE (33 PENCERE-BAĞIMSIZ ÖZELLİK)
    # =====================================================================
    print("\n--- [2/3] 8-SINIF BASELINE (33 Temiz Özellik, OvR AUC) ---")
    mc_rows = []
    mc_per_cls_all = []
    lgbm_imp_df = None

    for name, tag, scale_num, est in get_model_specs():
        pipe = build_feature_pipeline(est, scale_numeric=scale_num)
        t0 = time.perf_counter()
        pipe.fit(X_tr_33, y_tr_8)
        tr_sec = time.perf_counter() - t0

        y_pred = pipe.predict(X_va_33)
        y_prob = pipe.predict_proba(X_va_33)
        m_mc, per_cls_df = compute_multiclass_metrics(y_va_8, y_pred, y_prob, class_names=CLASS_8_ORDER)

        edge = measure_edge_cost(
            pipeline=pipe,
            X_eval=X_va_33,
            save_pipeline_path=mod_dir / f"exp1_mc8_{tag}_pipe.joblib",
            save_model_only_path=mod_dir / f"exp1_mc8_{tag}_model.joblib",
            train_time_sec=tr_sec,
            n_lat_samples=n_lat,
        )

        mc_rows.append({"model": name, "n_features": len(FEATURES_33), **m_mc, **edge})
        per_cls_df.insert(0, "model", name)
        mc_per_cls_all.append(per_cls_df)

        if name == "LightGBM":
            booster = pipe.named_steps["model"].booster_
            g = booster.feature_importance(importance_type="gain")
            s = booster.feature_importance(importance_type="split")
            lgbm_imp_df = pd.DataFrame({
                "feature": FEATURES_33,
                "mc8_gain_norm": np.round(g / g.sum(), 6),
                "mc8_split_norm": np.round(s / s.sum(), 6),
            })

        print(
            f"  [MC-8] {name:<20} | Macro-F1={m_mc['f1_macro']:.4f} | W-F1={m_mc['f1_weighted']:.4f} | "
            f"OvR-ROC={m_mc['roc_auc_ovr_macro']:.4f} | OvR-PR={m_mc['pr_auc_ovr_macro']:.4f} | "
            f"Size={edge['model_only_size_kb']:.1f}KB | RSS_d={edge['rss_model_delta_mb']}MB"
        )

    # =====================================================================
    # 3) HEDEFLİ ABLATION (LightGBM: All_39 vs Clean_33 vs Clean_32_NoHTTPS)
    # =====================================================================
    print("\n--- [3/3] HEDEFLİ ABLATION (LightGBM: Number Artefaktı & HTTPS Kontrolü) ---")
    ablation_regimes = [
        ("A0_All_39_WithWindowArtifact", ALL_39_FEATURES),
        ("A1_Clean_33_WindowIndependent", FEATURES_33),
        ("A2_Clean_32_NoHTTPS", FEATURES_32_NO_HTTPS),
    ]
    abl_rows = []

    for rname, cols in ablation_regimes:
        Xtr, Xva = df_tr[cols], df_va[cols]

        # Binary
        pb = build_feature_pipeline(
            LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            scale_numeric=False,
        )
        pb.fit(Xtr, y_tr_bin)
        ypr_b = pb.predict_proba(Xva)[:, 1]
        mb_def = compute_binary_metrics(y_va_bin, (ypr_b >= 0.50).astype(np.int8), ypr_b)
        thr_b, _ = find_best_binary_threshold(y_va_bin, ypr_b)
        mb_tun = compute_binary_metrics(y_va_bin, (ypr_b >= thr_b).astype(np.int8), ypr_b)

        gains_b = pb.named_steps["model"].booster_.feature_importance(importance_type="gain")
        splits_b = pb.named_steps["model"].booster_.feature_importance(importance_type="split")
        g_norm = gains_b / gains_b.sum()
        top5 = ", ".join(f"{c}({v:.2f})" for c, v in sorted(zip(cols, g_norm), key=lambda x: x[1], reverse=True)[:5])

        if rname == "A1_Clean_33_WindowIndependent" and lgbm_imp_df is not None:
            bin_imp = pd.DataFrame({
                "feature": cols,
                "bin_gain_norm": np.round(g_norm, 6),
                "bin_split_norm": np.round(splits_b / splits_b.sum(), 6),
            })
            lgbm_imp_df = bin_imp.merge(lgbm_imp_df, on="feature", how="left")
            lgbm_imp_df["combined_rank_score"] = np.round(
                0.5 * lgbm_imp_df["bin_gain_norm"] + 0.5 * lgbm_imp_df["mc8_gain_norm"], 6
            )
            lgbm_imp_df = lgbm_imp_df.sort_values("combined_rank_score", ascending=False).reset_index(drop=True)

        # 8-Class
        p8 = build_feature_pipeline(
            LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            scale_numeric=False,
        )
        p8.fit(Xtr, y_tr_8)
        m8, _ = compute_multiclass_metrics(y_va_8, p8.predict(Xva), p8.predict_proba(Xva), class_names=CLASS_8_ORDER)

        abl_rows.append({
            "regime": rname,
            "n_features": len(cols),
            "bin_def_f1_macro": mb_def["f1_macro"],
            "bin_tuned_thr": thr_b,
            "bin_tuned_f1_macro": mb_tun["f1_macro"],
            "bin_tuned_fpr": mb_tun["fpr"],
            "bin_tuned_recall": mb_tun["attack_recall"],
            "bin_roc_auc": mb_def["roc_auc"],
            "bin_pr_auc": mb_def["pr_auc"],
            "mc8_f1_macro": m8["f1_macro"],
            "mc8_f1_weighted": m8["f1_weighted"],
            "mc8_ovr_roc": m8["roc_auc_ovr_macro"],
            "bin_top5_gain": top5,
        })
        print(
            f"  [Abl ] {rname:<30} | Bin Tuned F1={mb_tun['f1_macro']:.4f} (FPR={mb_tun['fpr']:.4f}, ROC={mb_def['roc_auc']:.5f}) | "
            f"MC8 F1={m8['f1_macro']:.4f} | Top5: {top5}"
        )

    # --- KAYITLAR ---
    bin_df = pd.DataFrame(bin_rows).sort_values("tuned_f1_macro", ascending=False)
    mc_df = pd.DataFrame(mc_rows).sort_values("f1_macro", ascending=False)
    mc_per_cls_df = pd.concat(mc_per_cls_all, ignore_index=True)
    abl_df = pd.DataFrame(abl_rows)

    bin_df.to_csv(exp_dir / "exp1_binary_results.csv", index=False)
    mc_df.to_csv(exp_dir / "exp1_multiclass8_results.csv", index=False)
    mc_per_cls_df.to_csv(exp_dir / "exp1_multiclass8_per_class.csv", index=False)
    abl_df.to_csv(exp_dir / "exp1_ablation_window_https.csv", index=False)
    if lgbm_imp_df is not None:
        lgbm_imp_df.to_csv(exp_dir / "exp1_lgbm_feature_importance.csv", index=False)

    with open(exp_dir / "exp1_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "is_quick": bool(args.quick),
                "train_files": TRAIN_FILES,
                "train_frac": tr_frac,
                "train_rows": int(len(df_tr)),
                "val_files": EXP1_VAL_FILES,
                "val_rows": int(len(df_va)),
                "locked_test_files": TEST_FILES_LOCKED,
                "collinearity_check": collin,
                "hardware": hw,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== LIGHTGBM CLEAN_33 — EN ÖNEMLİ 15 ÖZELLİK (BINARY + 8-SINIF BİRLEŞİK GAIN) ===")
    if lgbm_imp_df is not None:
        print(lgbm_imp_df.head(15).to_string(index=False))

    print(f"\nExp 1 tamamlandı ({time.perf_counter() - t_all:.1f} sn). Çıktılar -> {exp_dir}")


if __name__ == "__main__":
    main()