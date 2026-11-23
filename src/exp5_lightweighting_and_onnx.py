import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from lightgbm import LGBMClassifier
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
    TRAIN_FILES,
    map_label_to_8_classes,
)

try:
    from src.config import FEATURES_TOP10, FEATURES_TOP15
except ImportError:
    FEATURES_TOP15 = [
        "HTTPS", "AVG", "IAT", "Header_Length", "ack_flag_number",
        "Max", "syn_flag_number", "Time_To_Live", "Min", "psh_flag_number",
        "rst_flag_number", "fin_flag_number", "Std", "HTTP", "ARP",
    ]
    FEATURES_TOP10 = FEATURES_TOP15[:10]

warnings.filterwarnings("ignore", category=UserWarning)

CAL_FILE = "Merged07.csv"
EVAL_FILE = "Merged08.csv"

CHECKPOINTS = [
    ("Top15", FEATURES_TOP15),
    ("Top10", FEATURES_TOP10),
]

# Yapısal ağaç budama kademeleri (daha az ağaçta yakınsama için learning_rate orantılı ayarlanır)
PRUNING_TIERS = [
    {"tier": "L0_Baseline",   "n_estimators": 150, "num_leaves": 63, "max_depth": -1, "learning_rate": 0.05},
    {"tier": "L1_Balanced",   "n_estimators": 80,  "num_leaves": 31, "max_depth": 7,  "learning_rate": 0.07},
    {"tier": "L2_Light",      "n_estimators": 50,  "num_leaves": 15, "max_depth": 6,  "learning_rate": 0.08},
    {"tier": "L3_UltraLight", "n_estimators": 30,  "num_leaves": 15, "max_depth": 5,  "learning_rate": 0.10},
    {"tier": "L4_Micro",      "n_estimators": 20,  "num_leaves": 7,  "max_depth": 4,  "learning_rate": 0.12},
]


# İzole süreçte saf numpy + onnxruntime RSS bellek ölçümü (pandas/sklearn/lightgbm YOK)
_ONNX_RSS_PROBE = r"""
import ctypes, gc, json, sys
import numpy as np
import onnxruntime as ort

def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 ** 2), "psutil_rss"
    except Exception:
        pass
    if sys.platform == "win32":
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / (1024.0 ** 2), "win32_WorkingSetSize"
    return None, "unavailable"

onnx_path, sample_npy = sys.argv[1], sys.argv[2]
sample = np.load(sample_npy).astype(np.float32)
gc.collect()
base_rss, method = rss_mb()

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
gc.collect()
load_rss, _ = rss_mb()

for i in range(min(30, len(sample))):
    sess.run(None, {inp_name: sample[i : i + 1]})
gc.collect()
infer_rss, _ = rss_mb()

print(json.dumps({
    "onnx_rss_method": method,
    "onnx_rss_baseline_mb": round(base_rss, 3) if base_rss is not None else None,
    "onnx_rss_model_delta_mb": round(max(0.0, load_rss - base_rss), 3) if (base_rss is not None and load_rss is not None) else None,
    "onnx_rss_total_after_infer_mb": round(infer_rss, 3) if infer_rss is not None else None,
}))
"""

# İzole süreçte sklearn + pandas + lightgbm pipeline RSS bellek ölçümü
_SKLEARN_RSS_PROBE = r"""
import ctypes, gc, json, sys
import joblib
try:
    import lightgbm
except Exception:
    pass

def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 ** 2), "psutil_rss"
    except Exception:
        pass
    if sys.platform == "win32":
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / (1024.0 ** 2), "win32_WorkingSetSize"
    return None, "unavailable"

pipe_path, sample_path = sys.argv[1], sys.argv[2]
sample = joblib.load(sample_path)
gc.collect()
base_rss, _ = rss_mb()
pipe = joblib.load(pipe_path)
for i in range(min(30, len(sample))):
    pipe.predict_proba(sample.iloc[i : i + 1])
gc.collect()
infer_rss, _ = rss_mb()

print(json.dumps({
    "sklearn_rss_total_after_infer_mb": round(infer_rss, 3) if infer_rss is not None else None,
}))
"""


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


def preprocess_numpy_fast(arr: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Canlı Edge çıkarımı için pandas/sklearn olmadan hızlı float32 + Inf/NaN doldurma."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    bad = ~np.isfinite(x)
    if bad.any():
        x = x.copy()
        _, cols = np.where(bad)
        x[bad] = medians[cols]
    return x


def export_lgbm_to_onnx(lgbm_model: LGBMClassifier, n_features: int, out_path: Path) -> None:
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    try:
        onnx_model = convert_lightgbm(
            lgbm_model,
            initial_types=initial_type,
            zipmap=False,
        )
    except TypeError:
        onnx_model = convert_lightgbm(
            lgbm_model,
            initial_types=initial_type,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())


def extract_ort_pos_prob(ort_outputs) -> np.ndarray:
    probs_raw = ort_outputs[1]
    if isinstance(probs_raw, np.ndarray):
        return probs_raw[:, 1].astype(np.float64)
    # ZipMap dict listesi fallback'i
    return np.array([d.get(1, d.get("1", 0.0)) for d in probs_raw], dtype=np.float64)


def run_isolated_rss_probes(onnx_path: Path, pipe_path: Path, X_sample_df: pd.DataFrame, X_sample_np: np.ndarray) -> dict:
    res = {
        "onnx_rss_method": "error",
        "onnx_rss_baseline_mb": np.nan,
        "onnx_rss_model_delta_mb": np.nan,
        "onnx_rss_total_after_infer_mb": np.nan,
        "sklearn_rss_total_after_infer_mb": np.nan,
    }
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf_np:
        np_path = Path(tf_np.name)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf_df:
        df_path = Path(tf_df.name)

    try:
        np.save(np_path, X_sample_np[:30].astype(np.float32))
        joblib.dump(X_sample_df.iloc[:30].copy(), df_path)

        p1 = subprocess.run(
            [sys.executable, "-c", _ONNX_RSS_PROBE, str(onnx_path), str(np_path)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if p1.returncode == 0 and p1.stdout.strip():
            res.update(json.loads(p1.stdout.strip().splitlines()[-1]))

        p2 = subprocess.run(
            [sys.executable, "-c", _SKLEARN_RSS_PROBE, str(pipe_path), str(df_path)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if p2.returncode == 0 and p2.stdout.strip():
            res.update(json.loads(p2.stdout.strip().splitlines()[-1]))
    finally:
        np_path.unlink(missing_ok=True)
        df_path.unlink(missing_ok=True)

    return res


def benchmark_sklearn_vs_onnx(
    pipe: Pipeline,
    lgbm_model: LGBMClassifier,
    medians: np.ndarray,
    X_ev_df: pd.DataFrame,
    X_ev_raw_np: np.ndarray,
    X_ev_imp_f32: np.ndarray,
    onnx_path: Path,
    pipe_path: Path,
    model_path: Path,
    n_lat: int = 300,
) -> tuple[np.ndarray, dict]:
    # Tek çekirdek zorlaması (adil Edge/IoT kıyası)
    lgbm_model.set_params(n_jobs=1)
    joblib.dump(pipe, pipe_path)
    joblib.dump(lgbm_model, model_path)

    joblib_model_kb = model_path.stat().st_size / 1024.0
    onnx_kb = onnx_path.stat().st_size / 1024.0

    # ONNX Runtime tek çekirdek oturumu
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # Batch hız (LightGBM NumPy vs ONNX Runtime E2E)
    t0 = time.perf_counter()
    _ = lgbm_model.predict_proba(X_ev_imp_f32)
    lgbm_batch_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_fast = preprocess_numpy_fast(X_ev_raw_np, medians)
    ort_out_batch = sess.run(None, {inp_name: x_fast})
    onnx_batch_sec = time.perf_counter() - t0
    prob_onnx = extract_ort_pos_prob(ort_out_batch)

    # Tekil paket (batch=1) gecikmeleri: Sklearn+Pandas E2E vs LightGBM NumPy vs ONNX E2E
    n_s = min(n_lat, len(X_ev_df))
    single_dfs = [X_ev_df.iloc[i : i + 1] for i in range(n_s)]
    single_raw_nps = [X_ev_raw_np[i : i + 1] for i in range(n_s)]
    single_imp_f32s = [X_ev_imp_f32[i : i + 1] for i in range(n_s)]

    # Warm-up
    for i in range(min(20, n_s)):
        pipe.predict_proba(single_dfs[i])
        lgbm_model.predict_proba(single_imp_f32s[i])
        sess.run(None, {inp_name: preprocess_numpy_fast(single_raw_nps[i], medians)})

    sk_lat, lgb_lat, ort_lat = [], [], []
    for i in range(n_s):
        ts = time.perf_counter()
        pipe.predict_proba(single_dfs[i])
        sk_lat.append((time.perf_counter() - ts) * 1000.0)

        ts = time.perf_counter()
        lgbm_model.predict_proba(single_imp_f32s[i])
        lgb_lat.append((time.perf_counter() - ts) * 1000.0)

        ts = time.perf_counter()
        x1 = preprocess_numpy_fast(single_raw_nps[i], medians)
        sess.run(None, {inp_name: x1})
        ort_lat.append((time.perf_counter() - ts) * 1000.0)

    rss_info = run_isolated_rss_probes(onnx_path, pipe_path, X_ev_df, X_ev_imp_f32)

    n_rows = len(X_ev_df)
    metrics = {
        "joblib_size_kb": round(float(joblib_model_kb), 2),
        "onnx_size_kb": round(float(onnx_kb), 2),
        "size_reduction_vs_joblib_%": round((1.0 - onnx_kb / joblib_model_kb) * 100.0, 1) if joblib_model_kb > 0 else 0.0,
        "sk_e2e_lat_p50_ms": round(float(np.percentile(sk_lat, 50)), 4),
        "sk_e2e_lat_p95_ms": round(float(np.percentile(sk_lat, 95)), 4),
        "lgb_np_lat_p50_ms": round(float(np.percentile(lgb_lat, 50)), 4),
        "onnx_e2e_lat_p50_ms": round(float(np.percentile(ort_lat, 50)), 4),
        "onnx_e2e_lat_p95_ms": round(float(np.percentile(ort_lat, 95)), 4),
        "onnx_e2e_lat_p99_ms": round(float(np.percentile(ort_lat, 99)), 4),
        "speedup_onnx_vs_sklearn_x": round(float(np.percentile(sk_lat, 50) / max(np.percentile(ort_lat, 50), 1e-6)), 1),
        "lgbm_throughput_pps": round(float(n_rows / lgbm_batch_sec), 1) if lgbm_batch_sec > 0 else np.nan,
        "onnx_throughput_pps": round(float(n_rows / onnx_batch_sec), 1) if onnx_batch_sec > 0 else np.nan,
        **rss_info,
    }
    return prob_onnx, metrics


def family_breakdown(df_eval: pd.DataFrame, prob: np.ndarray, thr: float) -> pd.DataFrame:
    preds = (prob >= thr).astype(np.int8)
    rows = []
    for cat in CLASS_8_ORDER:
        mask = df_eval["y_cat8"].values == cat
        tot = int(mask.sum())
        if cat == "Benign":
            fp = int((preds[mask] == 1).sum())
            rate = round((tot - fp) / tot * 100.0, 2) if tot else 0.0
            rows.append({"category": cat, "support": tot, "rate_pct": rate, "fp_or_missed": fp})
        else:
            tp = int((preds[mask] == 1).sum())
            rate = round(tp / tot * 100.0, 2) if tot else 0.0
            rows.append({"category": cat, "support": tot, "rate_pct": rate, "fp_or_missed": tot - tp})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Exp 5 — Yapısal Ağaç Budama + float32 & ONNX Runtime Hafifletme Deneyi (Top-15 & Top-10)"
    )
    parser.add_argument("--quick", action="store_true", help="Hızlı doğrulama modu (~25 sn).")
    args = parser.parse_args()

    t_all = time.perf_counter()
    hw = get_hardware_profile()

    assert CAL_FILE not in TRAIN_FILES and EVAL_FILE not in TRAIN_FILES
    assert CAL_FILE not in TEST_FILES_LOCKED and EVAL_FILE not in TEST_FILES_LOCKED

    if args.quick:
        exp_dir = RESULTS_DIR / "exp5_lightweighting_quick"
        mod_dir = MODELS_DIR / "exp5_lightweighting_quick"
        tr_files = TRAIN_FILES[:2]
        tr_ben, tr_atk, ev_frac, n_lat = 0.20, 0.03, 0.15, 50
    else:
        exp_dir = RESULTS_DIR / "exp5_lightweighting"
        mod_dir = MODELS_DIR / "exp5_lightweighting"
        tr_files = TRAIN_FILES
        tr_ben, tr_atk, ev_frac, n_lat = 1.00, 0.25, 1.00, 300

    exp_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================================")
    print(f"EXP 5 — AĞAÇ BUDAMA & ONNX RUNTIME HAFİFLETME ({'QUICK' if args.quick else 'TAM'})")
    print("=============================================================================")
    print(f"Checkpoints     : Top-15 ({len(FEATURES_TOP15)} oz.) & Top-10 ({len(FEATURES_TOP10)} oz.)")
    print(f"Train           : {tr_files} | Cal: {CAL_FILE} | Eval: {EVAL_FILE} | Test: {TEST_FILES_LOCKED} (KİLİTLİ)")
    print(f"Donanım         : {hw['os']} | CPU={hw['logical_cpu_count']} | RAM={hw['total_ram_gb']} GB\n")

    # Tüm Top-15 özelliklerini bir kez yükle (Top-10 zaten Top-15'in ilk 10'u)
    print("Veriler yükleniyor...")
    df_tr = load_files_ram_safe(tr_files, FEATURES_TOP15, benign_frac=tr_ben, attack_frac=tr_atk)
    df_cal = load_files_ram_safe([CAL_FILE], FEATURES_TOP15, benign_frac=ev_frac, attack_frac=ev_frac)
    df_eval = load_files_ram_safe([EVAL_FILE], FEATURES_TOP15, benign_frac=ev_frac, attack_frac=ev_frac)

    y_tr = df_tr["y_bin"].values
    y_cal = df_cal["y_bin"].values
    y_ev = df_eval["y_bin"].values
    print(f"Train: {len(df_tr):,} (Benign={(y_tr==0).sum():,}) | Cal: {len(df_cal):,} | Eval: {len(df_eval):,}\n")

    all_rows = []
    fam_frames = []
    deploy_configs = {}

    for ckpt_name, cols in CHECKPOINTS:
        print(f"==================== CHECKPOINT: {ckpt_name} ({len(cols)} ÖZELLİK) ====================")
        X_tr_df = df_tr[cols]
        X_cal_df = df_cal[cols]
        X_ev_df = df_eval[cols]

        # Yalnızca X_train üzerinde medyan öğren (sıfır sızıntı)
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_df).astype(np.float32)
        medians = imputer.statistics_.astype(np.float32)

        X_cal_imp = preprocess_numpy_fast(X_cal_df.to_numpy(dtype=np.float32), medians)
        X_ev_raw_np = X_ev_df.to_numpy(dtype=np.float32)
        X_ev_imp = preprocess_numpy_fast(X_ev_raw_np, medians)

        for cfg in PRUNING_TIERS:
            tier = cfg["tier"]
            tag = f"{ckpt_name.lower()}_{tier.lower()}"

            clf = LGBMClassifier(
                n_estimators=cfg["n_estimators"],
                num_leaves=cfg["num_leaves"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                class_weight=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            t0 = time.perf_counter()
            clf.fit(X_tr_imp, y_tr)
            tr_sec = time.perf_counter() - t0

            # Sklearn karşılaştırması için fit edilmiş Pipeline nesnesi
            pipe = Pipeline([("preprocessor", imputer), ("model", clf)])

            # ONNX dönüşümü
            onnx_path = mod_dir / f"{tag}.onnx"
            pipe_path = mod_dir / f"{tag}_pipe.joblib"
            model_path = mod_dir / f"{tag}_model.joblib"
            export_lgbm_to_onnx(clf, len(cols), onnx_path)

            # LightGBM olasılıkları (Merged07 kalibrasyon + Merged08 değerlendirme)
            prob_cal_lgb = clf.predict_proba(X_cal_imp)[:, 1]
            prob_ev_lgb = clf.predict_proba(X_ev_imp)[:, 1]

            # ONNX benchmarkı + ONNX Merged08 olasılıkları
            prob_ev_onnx, edge = benchmark_sklearn_vs_onnx(
                pipe=pipe,
                lgbm_model=clf,
                medians=medians,
                X_ev_df=X_ev_df,
                X_ev_raw_np=X_ev_raw_np,
                X_ev_imp_f32=X_ev_imp,
                onnx_path=onnx_path,
                pipe_path=pipe_path,
                model_path=model_path,
                n_lat=n_lat,
            )

            # ONNX eşdeğerlik doğrulaması (Merged07'de de ONNX kalibrasyonu kontrolü için aynı eşikler)
            prob_diff = np.abs(prob_ev_lgb - prob_ev_onnx)
            max_prob_diff = float(np.max(prob_diff))
            mean_prob_diff = float(np.mean(prob_diff))

            # Eşik kalibrasyonu (Merged07 üzerinde)
            thr_05 = threshold_for_target_fpr(y_cal, prob_cal_lgb, 0.005)
            thr_02 = threshold_for_target_fpr(y_cal, prob_cal_lgb, 0.002)
            thr_01 = threshold_for_target_fpr(y_cal, prob_cal_lgb, 0.001)

            # Merged08 üzerinde ONNX tahminleri ile nihai performans (Edge'de çalışacak gerçek çıktı!)
            pred_05 = (prob_ev_onnx >= thr_05).astype(np.int8)
            pred_02 = (prob_ev_onnx >= thr_02).astype(np.int8)
            pred_01 = (prob_ev_onnx >= thr_01).astype(np.int8)

            pred_01_lgb = (prob_ev_lgb >= thr_01).astype(np.int8)
            agree_01_pct = float((pred_01 == pred_01_lgb).mean() * 100.0)

            m05 = compute_binary_metrics(y_ev, pred_05, prob_ev_onnx)
            m02 = compute_binary_metrics(y_ev, pred_02, prob_ev_onnx)
            m01 = compute_binary_metrics(y_ev, pred_01, prob_ev_onnx)

            l99_p05, l99_f05 = compute_99pct_normal_metrics(y_ev, pred_05)
            l99_p02, l99_f02 = compute_99pct_normal_metrics(y_ev, pred_02)
            l99_p01, l99_f01 = compute_99pct_normal_metrics(y_ev, pred_01)

            # Dağıtım yapılandırma JSON'ı kaydet (her ONNX modeli kendi medyan ve eşiklerini taşır)
            cfg_json = {
                "checkpoint": ckpt_name,
                "tier": tier,
                "features": cols,
                "medians_float32": [float(v) for v in medians],
                "hyperparams": cfg,
                "thresholds_cal_m07": {
                    "target_fpr_0.5%": round(thr_05, 6),
                    "target_fpr_0.2%": round(thr_02, 6),
                    "target_fpr_0.1%": round(thr_01, 6),
                },
                "onnx_file": onnx_path.name,
            }
            deploy_configs[tag] = cfg_json
            (mod_dir / f"{tag}_config.json").write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")

            # Saldırı ailesi kırılımı (@0.1% ve @0.2% FPR)
            for b_lbl, thr_v in [("0.2%", thr_02), ("0.1%", thr_01)]:
                fdf = family_breakdown(df_eval, prob_ev_onnx, thr_v)
                fdf.insert(0, "target_fpr", b_lbl)
                fdf.insert(0, "tier", tier)
                fdf.insert(0, "checkpoint", ckpt_name)
                fam_frames.append(fdf)

            row = {
                "checkpoint": ckpt_name,
                "n_features": len(cols),
                "tier": tier,
                "n_estimators": cfg["n_estimators"],
                "num_leaves": cfg["num_leaves"],
                "max_depth": cfg["max_depth"],
                "train_sec": round(tr_sec, 2),
                "roc_auc": m01["roc_auc"],
                "pr_auc": m01["pr_auc"],
                "fpr05_real_%": round(m05["fpr"] * 100.0, 3),
                "fpr05_rec_%": round(m05["attack_recall"] * 100.0, 3),
                "fpr05_live99_p": l99_p05,
                "fpr02_real_%": round(m02["fpr"] * 100.0, 3),
                "fpr02_rec_%": round(m02["attack_recall"] * 100.0, 3),
                "fpr02_live99_p": l99_p02,
                "fpr02_live99_f1": l99_f02,
                "fpr01_real_%": round(m01["fpr"] * 100.0, 3),
                "fpr01_fp": m01["fp"],
                "fpr01_fn": m01["fn"],
                "fpr01_rec_%": round(m01["attack_recall"] * 100.0, 3),
                "fpr01_live99_p": l99_p01,
                "fpr01_live99_f1": l99_f01,
                "onnx_max_prob_diff": round(max_prob_diff, 7),
                "onnx_mean_prob_diff": round(mean_prob_diff, 7),
                "onnx_pred_agree_01_%": round(agree_01_pct, 5),
                **edge,
            }
            all_rows.append(row)

            print(
                f"  {tier:<14} | ROC={row['roc_auc']:.5f} | "
                f"@0.1%: FPR={row['fpr01_real_%']:.3f}% Rec={row['fpr01_rec_%']:.2f}% Live99P={l99_p01:.3f} | "
                f"Size(J/O)={edge['joblib_size_kb']:6.1f}/{edge['onnx_size_kb']:5.1f}KB | "
                f"Lat P50(Sk/ORT)={edge['sk_e2e_lat_p50_ms']:.3f}/{edge['onnx_e2e_lat_p50_ms']:.4f}ms ({edge['speedup_onnx_vs_sklearn_x']:.1f}x) | "
                f"RAM(Sk/ORT)={edge['sklearn_rss_total_after_infer_mb']}/{edge['onnx_rss_total_after_infer_mb']}MB | "
                f"Agree={agree_01_pct:.3f}%"
            )

            # Disk temizliği: büyük ara joblib dosyalarını bırakmaya gerek yok (CSV + ONNX zaten kayıtlı)
            pipe_path.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)

        print()

    res_df = pd.DataFrame(all_rows)
    fam_df = pd.concat(fam_frames, ignore_index=True)

    # Özet tablo
    res_df.to_csv(exp_dir / "exp5_pruning_and_onnx_results.csv", index=False)
    fam_df.to_csv(exp_dir / "exp5_family_breakdown.csv", index=False)
    with open(exp_dir / "exp5_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "is_quick": bool(args.quick),
                "checkpoints": {k: v for k, v in CHECKPOINTS},
                "pruning_tiers": PRUNING_TIERS,
                "train_files": tr_files,
                "cal_file": CAL_FILE,
                "eval_file": EVAL_FILE,
                "locked_test_files": TEST_FILES_LOCKED,
                "hardware": hw,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not args.quick:
        view_cols = [
            "checkpoint", "tier", "n_estimators", "num_leaves", "max_depth",
            "roc_auc", "fpr02_real_%", "fpr02_rec_%", "fpr02_live99_p",
            "fpr01_real_%", "fpr01_rec_%", "fpr01_live99_p",
            "joblib_size_kb", "onnx_size_kb",
            "sk_e2e_lat_p50_ms", "onnx_e2e_lat_p50_ms", "onnx_e2e_lat_p95_ms",
            "speedup_onnx_vs_sklearn_x",
            "sklearn_rss_total_after_infer_mb", "onnx_rss_total_after_infer_mb",
            "onnx_pred_agree_01_%",
        ]
        md_lines = [
            "# Exp 5 — Yapısal Ağaç Budama ve ONNX Runtime Hafifletme Sonuçları\n",
            f"- **Eğitim:** `Merged01–04` (`%100 Benign + %25 Saldırı`, uniform ağırlık)",
            f"- **Eşik Kalibrasyonu:** `{CAL_FILE}` | **Değerlendirme & Benchmark:** `{EVAL_FILE}` | **Kilitli Test:** `{TEST_FILES_LOCKED}`",
            f"- **Tek Çekirdek Kıyaslaması:** Hem LightGBM hem ONNX Runtime `1 thread` ile ölçülmüştür.\n",
            "## 1. Budama ve ONNX Dönüşümü Karşılaştırma Tablosu (`Merged08`)\n",
            res_df[view_cols].to_markdown(index=False),
            "\n## 2. Tam Metrik Dosyaları\n",
            "- `results/exp5_lightweighting/exp5_pruning_and_onnx_results.csv`",
            "- `results/exp5_lightweighting/exp5_family_breakdown.csv`",
            "- `models/exp5_lightweighting/*.onnx` ve `*_config.json`",
        ]
        notes_path = DOCS_DIR / "05_exp5_lightweighting_onnx_notes.md"
        notes_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"Rapor notları -> {notes_path}")

    print(f"Exp 5 tamamlandı ({time.perf_counter() - t_all:.1f} sn). Çıktılar -> {exp_dir}")


if __name__ == "__main__":
    main()