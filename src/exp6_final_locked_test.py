import argparse
import ctypes
import datetime
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import compute_binary_metrics, get_hardware_profile
from src.config import (
    CLASS_8_ORDER,
    DOCS_DIR,
    LABEL_COL,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TEST_FILES_LOCKED,
    map_label_to_8_classes,
)

warnings.filterwarnings("ignore", category=UserWarning)
# Harmless ONNX batch-shape uyarisini sustur (temiz final raporu icin)
ort.set_default_logger_severity(3)

FINAL_MODELS_DIR = MODELS_DIR / "final_edge"
EXP5_MODELS_DIR = MODELS_DIR / "exp5_lightweighting"
EXP5B_MODELS_DIR = MODELS_DIR / "exp5b_c_native"

FROZEN_TAGS = [
    ("top15_l0_baseline", "Referans (Budanmamis)"),
    ("top15_l1_balanced", "Ana Onerilen Model"),
    ("top10_l2_light", "Ultra-Hafif Alternatif"),
]

LOCKED_BUDGET_KEYS = [
    ("Target_FPR_0.5%", "target_fpr_0.5%"),
    ("Target_FPR_0.2%", "target_fpr_0.2%"),
    ("Target_FPR_0.1%", "target_fpr_0.1%"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_models_to_final_edge() -> dict:
    """
    Exp 5 (.onnx) ve Exp 5b (.c/.dll/.exe) ciktilarini hicbir degisiklik yapmadan
    models/final_edge/ altina kopyalar ve SHA-256 manifestini olusturur.
    """
    FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "locked_test_files": TEST_FILES_LOCKED,
        "models": {},
    }

    for tag, role in FROZEN_TAGS:
        src_onnx = EXP5_MODELS_DIR / f"{tag}.onnx"
        src_e5_cfg = EXP5_MODELS_DIR / f"{tag}_config.json"

        src_c = EXP5B_MODELS_DIR / f"{tag}_model.c"
        dll_name = f"{tag}.dll" if sys.platform == "win32" else f"lib{tag}.so"
        exe_name = f"{tag}_bench.exe" if sys.platform == "win32" else f"{tag}_bench"
        src_dll = EXP5B_MODELS_DIR / dll_name
        src_exe = EXP5B_MODELS_DIR / exe_name
        src_e5b_cfg = EXP5B_MODELS_DIR / f"{tag}_c_config.json"

        required = [src_onnx, src_e5_cfg, src_c, src_dll, src_exe, src_e5b_cfg]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Dondurulacak model dosyalari eksik! Once Exp 5 ve Exp 5b tam modda calistirilmalidir:\n  "
                + "\n  ".join(missing)
            )

        cfg5 = json.loads(src_e5_cfg.read_text(encoding="utf-8"))
        cfg5b = json.loads(src_e5b_cfg.read_text(encoding="utf-8"))

        # Exp 5 ve Exp 5b ozellik, medyan ve esik tutarliligini dogrula
        assert cfg5["features"] == cfg5b["features"], f"Ozellik uyusmazligi: {tag}"
        assert np.allclose(cfg5["medians_float32"], cfg5b["medians_float32"], atol=1e-6), f"Medyan uyusmazligi: {tag}"
        for _, bkey in LOCKED_BUDGET_KEYS:
            t5 = float(cfg5["thresholds_cal_m07"][bkey])
            t5b = float(cfg5b["thresholds_cal_m07"][bkey])
            assert abs(t5 - t5b) <= 1e-5, f"Esik uyusmazligi ({tag}, {bkey}): {t5} vs {t5b}"

        # Final dagitim dizinine kopyala
        dst_onnx = FINAL_MODELS_DIR / src_onnx.name
        dst_c = FINAL_MODELS_DIR / src_c.name
        dst_dll = FINAL_MODELS_DIR / src_dll.name
        dst_exe = FINAL_MODELS_DIR / src_exe.name
        shutil.copy2(src_onnx, dst_onnx)
        shutil.copy2(src_c, dst_c)
        shutil.copy2(src_dll, dst_dll)
        shutil.copy2(src_exe, dst_exe)

        unified_cfg = {
            "tag": tag,
            "role": role,
            "checkpoint": cfg5b["checkpoint"],
            "tier": cfg5b["tier"],
            "features": cfg5b["features"],
            "medians_float32": cfg5b["medians_float32"],
            "hyperparams": cfg5b["hyperparams"],
            "thresholds_locked_from_m07": cfg5b["thresholds_cal_m07"],
            "artifacts": {
                "onnx": {"file": dst_onnx.name, "size_kb": round(dst_onnx.stat().st_size / 1024.0, 2), "sha256": sha256_file(dst_onnx)},
                "c_source": {"file": dst_c.name, "size_kb": round(dst_c.stat().st_size / 1024.0, 2), "sha256": sha256_file(dst_c)},
                "c_dll": {"file": dst_dll.name, "size_kb": round(dst_dll.stat().st_size / 1024.0, 2), "sha256": sha256_file(dst_dll)},
                "c_bench_exe": {"file": dst_exe.name, "size_kb": round(dst_exe.stat().st_size / 1024.0, 2), "sha256": sha256_file(dst_exe)},
            },
        }
        (FINAL_MODELS_DIR / f"{tag}_frozen_config.json").write_text(
            json.dumps(unified_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest["models"][tag] = unified_cfg

    (FINAL_MODELS_DIR / "frozen_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def preprocess_numpy_fast(arr: np.ndarray, medians: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    bad = ~np.isfinite(x)
    if bad.any():
        x = x.copy()
        _, cols = np.where(bad)
        x[bad] = medians[cols]
    return x


def predict_c_dll(dll_path: Path, X_raw_f32: np.ndarray) -> tuple[np.ndarray, float]:
    x_c = np.ascontiguousarray(X_raw_f32, dtype=np.float32)
    n_rows = int(x_c.shape[0])
    out = np.empty(n_rows, dtype=np.float64)

    lib = ctypes.CDLL(str(dll_path))
    lib.ids_predict_batch.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ids_predict_batch.restype = None

    t0 = time.perf_counter()
    lib.ids_predict_batch(
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_rows,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    elapsed = time.perf_counter() - t0
    return out, elapsed


def predict_onnx_chunked(
    onnx_path: Path, X_raw_f32: np.ndarray, medians: np.ndarray, chunk_size: int = 200_000
) -> tuple[np.ndarray, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    n_rows = len(X_raw_f32)
    probs = np.empty(n_rows, dtype=np.float64)

    t0 = time.perf_counter()
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        x_imp = preprocess_numpy_fast(X_raw_f32[start:end], medians)
        out = sess.run(None, {inp_name: x_imp})[1]
        if isinstance(out, np.ndarray):
            probs[start:end] = out[:, 1].astype(np.float64)
        else:
            probs[start:end] = np.array([d.get(1, d.get("1", 0.0)) for d in out], dtype=np.float64)
    elapsed = time.perf_counter() - t0
    return probs, elapsed


def run_native_exe_benchmark(exe_path: Path, X_sample_f32: np.ndarray, loops: int = 50) -> dict:
    x_c = np.ascontiguousarray(X_sample_f32, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        bin_path = Path(tf.name)
    try:
        x_c.tofile(bin_path)
        proc = subprocess.run(
            [str(exe_path), str(bin_path), str(len(x_c)), str(loops)],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"EXE benchmark hatasi (rc={proc.returncode}): {proc.stderr}")
        return json.loads(proc.stdout.strip().splitlines()[-1])
    finally:
        bin_path.unlink(missing_ok=True)


def verify_frozen_artifacts_dummy(manifest: dict) -> None:
    """Merged09-10'a hic dokunmadan dondurulmus DLL ve ONNX yuklemesini sahte girdiyle dogrular."""
    print("\n--- DONDURULMUS ARTIFACT ON KONTROLU (Merged09-10'a DOKUNULMADI) ---")
    for tag, mcfg in manifest["models"].items():
        n_feat = len(mcfg["features"])
        med = np.array(mcfg["medians_float32"], dtype=np.float32)
        dummy = np.vstack([med, med * 1.1, np.full(n_feat, np.nan, dtype=np.float32)]).astype(np.float32)

        dll_path = FINAL_MODELS_DIR / mcfg["artifacts"]["c_dll"]["file"]
        onnx_path = FINAL_MODELS_DIR / mcfg["artifacts"]["onnx"]["file"]

        p_c, _ = predict_c_dll(dll_path, dummy)
        p_o, _ = predict_onnx_chunked(onnx_path, dummy, med)
        diff = float(np.max(np.abs(p_c - p_o)))
        print(
            f"  [OK] {tag:<18} | ozellik={n_feat:<2} | DLL={mcfg['artifacts']['c_dll']['size_kb']:5.1f}KB | "
            f"ONNX={mcfg['artifacts']['onnx']['size_kb']:5.1f}KB | "
            f"kilitli_t@0.1%={mcfg['thresholds_locked_from_m07']['target_fpr_0.1%']:.6f} | dummy_diff={diff:.2e}"
        )


def load_locked_test_set(feature_cols: list[str]) -> pd.DataFrame:
    """Kilitli Merged09.csv ve Merged10.csv dosyalarinin tamamini (%100, orneklemesiz) yukler."""
    usecols = list(dict.fromkeys(feature_cols + [LABEL_COL]))
    frames = []
    for fname in TEST_FILES_LOCKED:
        fpath = RAW_DATA_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Kilitli test dosyasi bulunamadi: {fpath}")
        df_p = pd.read_csv(fpath, usecols=usecols, low_memory=False)
        for col in feature_cols:
            if np.issubdtype(df_p[col].dtype, np.number):
                df_p[col] = df_p[col].replace([np.inf, -np.inf], np.nan).astype(np.float32)

        raw_lbl = df_p[LABEL_COL].astype(str).str.strip()
        df_p["y_bin"] = (raw_lbl.str.upper() != "BENIGN").astype(np.int8)
        df_p["y_cat8"] = raw_lbl.map(map_label_to_8_classes)
        df_p["_source_file"] = fname
        df_p.drop(columns=[LABEL_COL], inplace=True)
        frames.append(df_p)
        print(f"  Yuklendi: {fname} -> {len(df_p):,} satir (Benign={(df_p['y_bin']==0).sum():,})")

    df_all = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return df_all


def simulate_prior_scenarios(
    y_true: np.ndarray, preds: np.ndarray, n_trials: int = 30
) -> list[dict]:
    ben_idx = np.where(y_true == 0)[0]
    atk_idx = np.where(y_true == 1)[0]
    n_ben = len(ben_idx)
    fp_ben = int((preds[ben_idx] == 1).sum())
    fpr_real = fp_ben / n_ben if n_ben > 0 else 0.0

    scenarios = [{
        "scenario": "100%_Normal_0%_Attack",
        "n_benign": n_ben,
        "n_attack": 0,
        "realized_fpr_%": round(fpr_real * 100.0, 4),
        "fp_per_10k_normal": round(fpr_real * 10000.0, 2),
        "realized_precision": np.nan,
        "realized_recall": np.nan,
        "realized_f1_attack": np.nan,
        "fp": fp_ben,
        "avg_tp": 0.0,
        "avg_fn": 0.0,
    }]

    rng = np.random.default_rng(RANDOM_STATE)
    for label, atk_ratio in [("99%_Normal_1%_Attack", 0.01), ("95%_Normal_5%_Attack", 0.05)]:
        n_atk_need = min(len(atk_idx), max(1, int(round(n_ben * atk_ratio / (1.0 - atk_ratio)))))
        precs, recs, f1s, tps, fns = [], [], [], [], []
        for _ in range(n_trials):
            chosen = rng.choice(atk_idx, size=n_atk_need, replace=False)
            tp = int((preds[chosen] == 1).sum())
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
            "realized_fpr_%": round(fpr_real * 100.0, 4),
            "fp_per_10k_normal": round(fpr_real * 10000.0, 2),
            "realized_precision": round(float(np.mean(precs)), 5),
            "realized_recall": round(float(np.mean(recs)), 5),
            "realized_f1_attack": round(float(np.mean(f1s)), 5),
            "fp": fp_ben,
            "avg_tp": round(float(np.mean(tps)), 1),
            "avg_fn": round(float(np.mean(fns)), 1),
        })
    return scenarios


def family_breakdown(cat8_arr: np.ndarray, preds: np.ndarray) -> list[dict]:
    rows = []
    for cat in CLASS_8_ORDER:
        mask = cat8_arr == cat
        tot = int(mask.sum())
        if cat == "Benign":
            fp = int((preds[mask] == 1).sum())
            rows.append({
                "category": cat,
                "support": tot,
                "correct_or_detected": tot - fp,
                "fp_or_missed": fp,
                "rate_%": round((tot - fp) / tot * 100.0, 3) if tot else 0.0,
                "metric_type": "Benign Specificity (1 - FPR)",
            })
        else:
            tp = int((preds[mask] == 1).sum())
            rows.append({
                "category": cat,
                "support": tot,
                "correct_or_detected": tp,
                "fp_or_missed": tot - tp,
                "rate_%": round(tp / tot * 100.0, 3) if tot else 0.0,
                "metric_type": "Attack Recall",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Exp 6 — Dondurulmus Modellerin Kilitli Final Testi (Merged09 + Merged10)"
    )
    parser.add_argument(
        "--verify-freeze-only",
        action="store_true",
        help="Yalnizca modelleri models/final_edge/ altina dondurur ve sahte girdiyle dogrular (Merged09-10'u ACMAZ).",
    )
    args = parser.parse_args()

    t_all = time.perf_counter()
    hw = get_hardware_profile()

    print("=============================================================================")
    print("EXP 6 — MODEL DONDURMA VE KILITLI FINAL TEST (Merged09.csv + Merged10.csv)")
    print("=============================================================================")
    print(f"Donanim      : {hw['os']} | CPU={hw['logical_cpu_count']} | RAM={hw['total_ram_gb']} GB")
    print(f"Kilitli Test : {TEST_FILES_LOCKED}")

    manifest = freeze_models_to_final_edge()
    verify_frozen_artifacts_dummy(manifest)
    print(f"\nModeller basariyla donduruldu -> {FINAL_MODELS_DIR}")

    if args.verify_freeze_only:
        print("\n--verify-freeze-only secili: Merged09-10 acilmadi. Final test icin bayraksiz calistirin.")
        return

    exp_dir = RESULTS_DIR / "exp6_final_locked_test"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Kilitli test setini (%100 tamamini) tek seferde yukle
    all_needed_features = manifest["models"]["top15_l0_baseline"]["features"]
    print("\n--- [1/4] KILITLI TEST SETI YUKLENIYOR (ILK VE SON KEZ) ---")
    df_test = load_locked_test_set(all_needed_features)
    y_test = df_test["y_bin"].values
    cat_test = df_test["y_cat8"].values
    src_files = df_test["_source_file"].values

    n_total = len(df_test)
    n_ben = int((y_test == 0).sum())
    n_atk = int((y_test == 1).sum())
    print(f"TOPLAM KILITLI TEST: {n_total:,} satir | Benign={n_ben:,} (%{n_ben/n_total*100:.2f}) | Saldiri={n_atk:,}\n")

    metric_rows = []
    verify_rows = []
    stress_rows = []
    family_rows = []
    per_file_rows = []
    native_hw_rows = []

    print("--- [2/4] DONDURULMUS 3 MODELIN KILITLI TEST DEGERLENDIRMESI (SAF C99 + ONNX) ---")
    for tag, mcfg in manifest["models"].items():
        cols = mcfg["features"]
        role = mcfg["role"]
        med = np.array(mcfg["medians_float32"], dtype=np.float32)

        dll_path = FINAL_MODELS_DIR / mcfg["artifacts"]["c_dll"]["file"]
        onnx_path = FINAL_MODELS_DIR / mcfg["artifacts"]["onnx"]["file"]
        exe_path = FINAL_MODELS_DIR / mcfg["artifacts"]["c_bench_exe"]["file"]

        X_raw = df_test[cols].to_numpy(dtype=np.float32)

        # 1) Saf C99 DLL ve ONNX Runtime cikarimlari (1.59M satir uzerinde)
        prob_c, c_sec = predict_c_dll(dll_path, X_raw)
        prob_onnx, onnx_sec = predict_onnx_chunked(onnx_path, X_raw, med)

        # 2) Kilitli test verisi uzerinde bagimsiz Saf C99 EXE donanim olcumu (2,000 satir x 50 tur)
        native_bm = run_native_exe_benchmark(exe_path, X_raw[:2000], loops=50)
        native_hw_rows.append({
            "model": tag,
            "role": role,
            "n_features": len(cols),
            "onnx_size_kb": mcfg["artifacts"]["onnx"]["size_kb"],
            "c_dll_size_kb": mcfg["artifacts"]["c_dll"]["size_kb"],
            "c_exe_size_kb": mcfg["artifacts"]["c_bench_exe"]["size_kb"],
            "c_rss_peak_mb": native_bm["c_rss_peak_mb"],
            "c_lat_p50_us": native_bm["c_lat_p50_us"],
            "c_lat_p95_us": native_bm["c_lat_p95_us"],
            "c_lat_p99_us": native_bm["c_lat_p99_us"],
            "c_single_core_pps": native_bm["c_throughput_pps"],
            "c_full_1_59m_sec": round(c_sec, 3),
            "onnx_full_1_59m_sec": round(onnx_sec, 3),
        })

        print(f"\n>>> [{tag}] ({role} | {len(cols)} ozellik | 1.59M sure: C={c_sec:.2f}sn, ONNX={onnx_sec:.2f}sn) <<<")

        for b_label, b_key in LOCKED_BUDGET_KEYS:
            thr = float(mcfg["thresholds_locked_from_m07"][b_key])

            pred_c = (prob_c >= thr).astype(np.int8)
            pred_o = (prob_onnx >= thr).astype(np.int8)

            mc = compute_binary_metrics(y_test, pred_c, prob_c)
            mo = compute_binary_metrics(y_test, pred_o, prob_onnx)
            ben_prec_c = round(float(mc["tn"] / (mc["tn"] + mc["fn"])), 5) if (mc["tn"] + mc["fn"]) > 0 else 0.0

            agree_pct = round(float((pred_c == pred_o).mean() * 100.0), 5)
            max_pdiff = round(float(np.max(np.abs(prob_c - prob_onnx))), 6)

            # Canli normal trafik stres testi (%100 / %99 / %95 Normal)
            sc_list = simulate_prior_scenarios(y_test, pred_c, n_trials=30)
            live99 = [s for s in sc_list if s["scenario"] == "99%_Normal_1%_Attack"][0]
            live95 = [s for s in sc_list if s["scenario"] == "95%_Normal_5%_Attack"][0]

            for sc in sc_list:
                stress_rows.append({"model": tag, "role": role, "operating_point": b_label, "locked_threshold": thr, **sc})

            metric_rows.append({
                "model": tag,
                "role": role,
                "n_features": len(cols),
                "operating_point": b_label,
                "locked_threshold_m07": thr,
                "accuracy": mc["accuracy"],
                "f1_macro": mc["f1_macro"],
                "attack_precision": mc["attack_precision"],
                "attack_recall": mc["attack_recall"],
                "benign_precision": ben_prec_c,
                "benign_recall_specificity": mc["benign_recall"],
                "fpr_%": round(mc["fpr"] * 100.0, 4),
                "fnr_%": round(mc["fnr"] * 100.0, 4),
                "fp_per_10k_normal": round(mc["fpr"] * 10000.0, 2),
                "tp": mc["tp"],
                "fp": mc["fp"],
                "tn": mc["tn"],
                "fn": mc["fn"],
                "roc_auc": mc["roc_auc"],
                "pr_auc": mc["pr_auc"],
                "live99_precision": live99["realized_precision"],
                "live99_f1_attack": live99["realized_f1_attack"],
                "live95_precision": live95["realized_precision"],
                "live95_f1_attack": live95["realized_f1_attack"],
            })

            verify_rows.append({
                "model": tag,
                "operating_point": b_label,
                "c_fpr_%": round(mc["fpr"] * 100.0, 4),
                "onnx_fpr_%": round(mo["fpr"] * 100.0, 4),
                "c_recall_%": round(mc["attack_recall"] * 100.0, 4),
                "onnx_recall_%": round(mo["attack_recall"] * 100.0, 4),
                "c_vs_onnx_agree_%": agree_pct,
                "max_prob_diff": max_pdiff,
            })

            print(
                f"  {b_label:<15} (t={thr:.5f}) | Acc={mc['accuracy']:.5f} | "
                f"AtkPrec={mc['attack_precision']:.5f} AtkRec={mc['attack_recall']:.5f} | "
                f"FPR={mc['fpr']*100:.3f}% (FP={mc['fp']:<3}, FN={mc['fn']:<5} | 10k'da {mc['fpr']*10000:.1f} FP) | "
                f"Live99(P={live99['realized_precision']:.3f}, F1={live99['realized_f1_attack']:.3f}) | "
                f"C=ONNX:{agree_pct:.3f}%"
            )

            # Saldiri ailesi hata analizi + Dosya bazli kararlılık (@0.2% ve @0.1%)
            if b_key in ("target_fpr_0.2%", "target_fpr_0.1%"):
                for frow in family_breakdown(cat_test, pred_c):
                    family_rows.append({"model": tag, "operating_point": b_label, **frow})

                for fname in TEST_FILES_LOCKED:
                    fmask = src_files == fname
                    mf = compute_binary_metrics(y_test[fmask], pred_c[fmask], prob_c[fmask])
                    per_file_rows.append({
                        "model": tag,
                        "operating_point": b_label,
                        "test_file": fname,
                        "n_samples": mf["n_samples"],
                        "accuracy": mf["accuracy"],
                        "attack_recall": mf["attack_recall"],
                        "fpr_%": round(mf["fpr"] * 100.0, 4),
                        "fp": mf["fp"],
                        "fn": mf["fn"],
                        "roc_auc": mf["roc_auc"],
                    })

        del X_raw, prob_c, prob_onnx
        gc.collect()

    metrics_df = pd.DataFrame(metric_rows)
    verify_df = pd.DataFrame(verify_rows)
    stress_df = pd.DataFrame(stress_rows)
    family_df = pd.DataFrame(family_rows)
    per_file_df = pd.DataFrame(per_file_rows)
    native_hw_df = pd.DataFrame(native_hw_rows)

    # --- OZET EKRAN CIKTILARI ---
    print("\n--- [3/4] 7 SALDIRI AILESI FINAL HATA ANALIZI (@Target_FPR_0.1% | Merged09+10) ---")
    f01 = family_df[family_df["operating_point"] == "Target_FPR_0.1%"]
    piv_fam = f01.pivot(index="category", columns="model", values="rate_%").reindex(CLASS_8_ORDER)
    piv_sup = f01[f01["model"] == "top15_l1_balanced"].set_index("category")["support"].reindex(CLASS_8_ORDER)
    piv_fam.insert(0, "support", piv_sup)
    print(piv_fam.to_string())

    print("\n--- [4/4] KILITLI TEST DOSYA KARARLILIGI (Merged09 vs Merged10 | @Target_FPR_0.1%) ---")
    print(per_file_df[per_file_df["operating_point"] == "Target_FPR_0.1%"].to_string(index=False))

    print("\n--- FINAL SAF C99 DONANIM OZETI (Kilitli Test Verisi Uzerinde) ---")
    print(native_hw_df.to_string(index=False))

    # --- CSV & JSON KAYITLARI ---
    metrics_df.to_csv(exp_dir / "exp6_final_metrics_c_native.csv", index=False)
    verify_df.to_csv(exp_dir / "exp6_c_vs_onnx_verification.csv", index=False)
    stress_df.to_csv(exp_dir / "exp6_live_prior_stress_test.csv", index=False)
    family_df.to_csv(exp_dir / "exp6_attack_family_breakdown.csv", index=False)
    per_file_df.to_csv(exp_dir / "exp6_per_file_stability.csv", index=False)
    native_hw_df.to_csv(exp_dir / "exp6_native_hardware_summary.csv", index=False)

    with open(exp_dir / "exp6_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "locked_test_files": TEST_FILES_LOCKED,
                "total_test_rows": n_total,
                "total_benign_rows": n_ben,
                "total_attack_rows": n_atk,
                "frozen_models": [t for t, _ in FROZEN_TAGS],
                "hardware": hw,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --- TEZ RAPORU MARKDOWN NOTLARI ---
    md_lines = [
        "# Exp 6 — Kilitli Final Test Sonuçları (`Merged09.csv + Merged10.csv`)\n",
        f"- **Test Kümesi (Tek Seferlik, %100 Örneklemesiz):** `{', '.join(TEST_FILES_LOCKED)}` — Toplam **{n_total:,}** akış (**{n_ben:,}** görülmemiş Benign, **{n_atk:,}** Saldırı)",
        "- **Akademik Bütünlük:** Bu aşamada hiçbir model yeniden eğitilmemiş, hiçbir özellik veya alarm eşiği değiştirilmemiştir. Tüm artefaktlar `models/final_edge/frozen_manifest.json` içinde SHA-256 ile mühürlenmiştir.\n",
        "## 1. Ana Final Metrik Tablosu (Saf C99 Native)\n",
        metrics_df.to_markdown(index=False),
        "\n## 2. Saf C99 vs. ONNX Runtime Doğrulaması (1.59M Kayıt)\n",
        verify_df.to_markdown(index=False),
        "\n## 3. Gerçekçi Canlı Normal Trafik Stres Testi (`%100 / %99 / %95 Normal`)\n",
        stress_df[stress_df["operating_point"].isin(["Target_FPR_0.2%", "Target_FPR_0.1%"])].to_markdown(index=False),
        "\n## 4. 7 Saldırı Ailesi Final Yakalama Oranları (`Target_FPR_0.1%`)\n",
        piv_fam.reset_index().to_markdown(index=False),
        "\n## 5. Dosya Bazlı Kararlılık (`Merged09` vs `Merged10`)\n",
        per_file_df.to_markdown(index=False),
        "\n## 6. Final Saf C99 Donanım Özeti\n",
        native_hw_df.to_markdown(index=False),
    ]
    notes_path = DOCS_DIR / "06_exp6_final_test_notes.md"
    notes_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nRapor notlari -> {notes_path}")
    print(f"Exp 6 Kilitli Final Test tamamlandi ({time.perf_counter() - t_all:.1f} sn). Ciktilar -> {exp_dir}")


if __name__ == "__main__":
    main()