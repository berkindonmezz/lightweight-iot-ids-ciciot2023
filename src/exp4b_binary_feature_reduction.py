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
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import compute_binary_metrics, get_hardware_profile, measure_edge_cost
from src.config import (
    DOCS_DIR,
    FEATURES_33,
    LABEL_COL,
    MODELS_DIR,
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
K_LIST = [24, 20, 15, 12, 10, 8, 5]

WEIGHT_MODES = {
    "uniform": {"Benign": 1.0, "DDoS": 1.0, "DoS": 1.0, "Mirai": 1.0, "Spoofing": 1.0, "Recon": 1.0, "BruteForce": 1.0, "Web": 1.0},
    "mild": {"Benign": 1.0, "DDoS": 1.0, "DoS": 1.0, "Mirai": 1.0, "Spoofing": 4.0, "Recon": 5.0, "BruteForce": 15.0, "Web": 15.0},
    "strong": {"Benign": 2.0, "DDoS": 1.0, "DoS": 1.0, "Mirai": 1.0, "Spoofing": 10.0, "Recon": 15.0, "BruteForce": 40.0, "Web": 40.0},
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


def stratified_subsample_idx(y: np.ndarray, max_n: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    if len(y) <= max_n:
        return np.arange(len(y))
    rng = np.random.default_rng(random_state)
    idx_list = []
    classes, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(classes, counts):
        c_idx = np.where(y == c)[0]
        take = min(len(c_idx), max(1, int(round(max_n * (cnt / len(y))))))
        idx_list.append(rng.choice(c_idx, size=take, replace=False))
    return np.concatenate(idx_list)


def main():
    parser = argparse.ArgumentParser(
        description="Exp 4b — Binary LightGBM Özellik Daraltma (24 -> 5) & Edge Maliyet Analizi"
    )
    parser.add_argument("--quick", action="store_true", help="Hızlı doğrulama modu (~20 sn).")
    parser.add_argument(
        "--weight-mode",
        choices=["uniform", "mild", "strong"],
        default="uniform",
        help="exp4a sonucuna göre eğitim ağırlık rejimi (varsayılan: uniform).",
    )
    args = parser.parse_args()

    t_all = time.perf_counter()
    hw = get_hardware_profile()

    assert CAL_FILE not in TRAIN_FILES and EVAL_FILE not in TRAIN_FILES
    assert CAL_FILE not in TEST_FILES_LOCKED and EVAL_FILE not in TEST_FILES_LOCKED

    if args.quick:
        exp_dir = RESULTS_DIR / "exp4b_feature_reduction_quick"
        mod_dir = MODELS_DIR / "exp4b_feature_reduction_quick"
        tr_files = TRAIN_FILES[:2]
        tr_ben, tr_atk, ev_frac, perm_n, n_lat = 0.20, 0.03, 0.15, 8000, 40
    else:
        exp_dir = RESULTS_DIR / "exp4b_feature_reduction"
        mod_dir = MODELS_DIR / "exp4b_feature_reduction"
        tr_files = TRAIN_FILES
        tr_ben, tr_atk, ev_frac, perm_n, n_lat = 1.00, 0.25, 1.00, 50000, 200

    exp_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================================")
    print(f"EXP 4B — BINARY ÖZELLİK DARALTMA (24 -> 5) | weight={args.weight_mode} ({'QUICK' if args.quick else 'TAM'})")
    print("=============================================================================")

    df_tr = load_files_ram_safe(tr_files, FEATURES_24, benign_frac=tr_ben, attack_frac=tr_atk)
    df_cal = load_files_ram_safe([CAL_FILE], FEATURES_24, benign_frac=ev_frac, attack_frac=ev_frac)
    df_eval = load_files_ram_safe([EVAL_FILE], FEATURES_24, benign_frac=ev_frac, attack_frac=ev_frac)

    y_tr = df_tr["y_bin"].values
    y_cal = df_cal["y_bin"].values
    y_ev = df_eval["y_bin"].values
    sw_tr = df_tr["y_cat8"].map(WEIGHT_MODES[args.weight_mode]).astype(np.float32).values

    # =========================================================================
    # 1) SIZINTISIZ ÖZELLİK SIRALAMASI (Yalnızca Train Gain + Merged07 PermImp)
    # =========================================================================
    print("--- [1/2] 24 ÖZELLİK SIRALAMASI (Train Gain + Merged07 Permutation Importance) ---")
    base_pipe = build_feature_pipeline(
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
    base_pipe.fit(df_tr[FEATURES_24], y_tr, model__sample_weight=sw_tr)

    gains = base_pipe.named_steps["model"].booster_.feature_importance(importance_type="gain")
    gains_n = gains / max(gains.sum(), 1e-12)

    sub_idx = stratified_subsample_idx(y_cal, perm_n, RANDOM_STATE)
    perm = permutation_importance(
        base_pipe,
        df_cal[FEATURES_24].iloc[sub_idx],
        y_cal[sub_idx],
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_pos = np.maximum(0.0, perm.importances_mean)
    perm_n = perm_pos / max(perm_pos.sum(), 1e-12)

    # Sıralama skoru: %70 Cal ROC-AUC Permutation Importance + %30 Train Gain
    rank_df = pd.DataFrame({
        "feature": FEATURES_24,
        "cal_perm_roc_auc": np.round(perm.importances_mean, 6),
        "cal_perm_norm": np.round(perm_n, 6),
        "train_gain_norm": np.round(gains_n, 6),
        "selection_score": np.round(0.70 * perm_n + 0.30 * gains_n, 6),
    }).sort_values("selection_score", ascending=False).reset_index(drop=True)
    rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))

    print(rank_df.to_string(index=False))
    ordered_features = rank_df["feature"].tolist()

    # =========================================================================
    # 2) TOP-K DARALTMA DÖNGÜSÜ (24 -> 20 -> 15 -> 12 -> 10 -> 8 -> 5)
    # =========================================================================
    print("\n--- [2/2] TOP-K PERFORMANS VE EDGE MALİYET KARŞILAŞTIRMASI (Eval: Merged08) ---")
    rows = []
    selected_sets = {}

    for k in K_LIST:
        cols_k = ordered_features[:k]
        selected_sets[f"top_{k}"] = cols_k

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
        pipe.fit(df_tr[cols_k], y_tr, model__sample_weight=sw_tr)
        tr_sec = time.perf_counter() - t0

        prob_cal = pipe.predict_proba(df_cal[cols_k])[:, 1]
        prob_ev = pipe.predict_proba(df_eval[cols_k])[:, 1]

        thr_05 = threshold_for_target_fpr(y_cal, prob_cal, 0.005)
        thr_01 = threshold_for_target_fpr(y_cal, prob_cal, 0.001)

        pred_05 = (prob_ev >= thr_05).astype(np.int8)
        pred_01 = (prob_ev >= thr_01).astype(np.int8)

        m05 = compute_binary_metrics(y_ev, pred_05, prob_ev)
        m01 = compute_binary_metrics(y_ev, pred_01, prob_ev)
        l99_p05, l99_f05 = compute_99pct_normal_metrics(y_ev, pred_05)
        l99_p01, l99_f01 = compute_99pct_normal_metrics(y_ev, pred_01)

        edge = measure_edge_cost(
            pipeline=pipe,
            X_eval=df_eval[cols_k],
            save_pipeline_path=mod_dir / f"lgbm_bin_top{k}_pipe.joblib",
            save_model_only_path=mod_dir / f"lgbm_bin_top{k}_model.joblib",
            train_time_sec=tr_sec,
            n_lat_samples=n_lat,
        )

        rows.append({
            "n_features": k,
            "roc_auc": m05["roc_auc"],
            "pr_auc": m05["pr_auc"],
            "fpr05_thr": round(thr_05, 5),
            "fpr05_real_fpr": m05["fpr"],
            "fpr05_fp": m05["fp"],
            "fpr05_fn": m05["fn"],
            "fpr05_atk_recall": m05["attack_recall"],
            "fpr05_live99_prec": l99_p05,
            "fpr05_live99_f1": l99_f05,
            "fpr01_thr": round(thr_01, 5),
            "fpr01_real_fpr": m01["fpr"],
            "fpr01_fp": m01["fp"],
            "fpr01_fn": m01["fn"],
            "fpr01_atk_recall": m01["attack_recall"],
            "fpr01_live99_prec": l99_p01,
            "fpr01_live99_f1": l99_f01,
            "model_size_kb": edge["model_only_size_kb"],
            "rss_delta_mb": edge["rss_model_delta_mb"],
            "lat_model_p50_ms": edge["model_only_latency_p50_ms"],
            "lat_e2e_p50_ms": edge["e2e_latency_p50_ms"],
            "throughput_e2e_pps": edge["throughput_e2e_pps"],
            "train_time_sec": edge["train_time_sec"],
        })

        print(
            f"  Top-{k:<2} | ROC={m05['roc_auc']:.5f} | "
            f"@0.5%: FPR={m05['fpr']*100:.3f}% Rec={m05['attack_recall']:.5f} Live99P={l99_p05:.3f} | "
            f"@0.1%: FPR={m01['fpr']*100:.3f}% Rec={m01['attack_recall']:.5f} Live99P={l99_p01:.3f} | "
            f"Size={edge['model_only_size_kb']:.1f}KB Lat(M/E2E)={edge['model_only_latency_p50_ms']:.3f}/{edge['e2e_latency_p50_ms']:.3f}ms"
        )

    res_df = pd.DataFrame(rows)

    rank_df.to_csv(exp_dir / "binary_24_feature_ranking.csv", index=False)
    res_df.to_csv(exp_dir / "feature_reduction_curve.csv", index=False)
    with open(exp_dir / "selected_feature_sets.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "is_quick": bool(args.quick),
                "weight_mode": args.weight_mode,
                "selected_sets": selected_sets,
                "hardware": hw,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not args.quick:
        md_lines = [
            f"# Exp 4b — Binary LightGBM Özellik Daraltma (24 → 5, weight={args.weight_mode})\n",
            "Özellik sırası yalnızca `Train` (Gain) + `Merged07` (ROC-AUC Permutation Importance) ile belirlenmiş, performans ve Edge maliyeti görülmemiş `Merged08` üzerinde ölçülmüştür.\n",
            "## 1. 24 Özellik Sıralaması\n",
            rank_df.to_markdown(index=False),
            "\n## 2. Özellik Sayısı vs. Performans ve Edge Maliyeti (`Merged08`)\n",
            res_df.to_markdown(index=False),
            "\n## 3. Aday Küçük Özellik Setleri\n",
            f"- **Top-15:** `{', '.join(selected_sets['top_15'])}`",
            f"- **Top-10:** `{', '.join(selected_sets['top_10'])}`",
            f"- **Top-8:** `{', '.join(selected_sets['top_8'])}`",
            f"- **Top-5:** `{', '.join(selected_sets['top_5'])}`",
        ]
        notes_path = DOCS_DIR / "exp4b_feature_reduction_notes.md"
        notes_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"\nRapor notları -> {notes_path}")

    print(f"Exp 4b tamamlandı ({time.perf_counter() - t_all:.1f} sn). Çıktılar -> {exp_dir}")


if __name__ == "__main__":
    main()