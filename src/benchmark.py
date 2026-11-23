import ctypes
import gc
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
try:
    import lightgbm
    _LGBM_VER = lightgbm.__version__
except Exception:
    _LGBM_VER = "not_installed"

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


def get_hardware_profile() -> dict:
    total_ram_gb = None
    if _HAS_PSUTIL:
        total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    elif sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_gb = round(stat.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            total_ram_gb = None

    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "total_ram_gb": total_ram_gb,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": _LGBM_VER,
    }


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    y_prob_pos = np.asarray(y_prob_pos, dtype=np.float64)

    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()]
    n_benign = tn + fp
    n_attack = tp + fn

    fpr = fp / n_benign if n_benign > 0 else np.nan
    fnr = fn / n_attack if n_attack > 0 else np.nan
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    has_both = len(np.unique(y_true)) == 2
    return {
        "task_type": "binary",
        "auc_scheme": "binary_positive_class_1_attack",
        "n_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 5),
        "benign_recall": round(float(rec[0]), 5),
        "attack_precision": round(float(prec[1]), 5),
        "attack_recall": round(float(rec[1]), 5),
        "attack_f1": round(float(f1[1]), 5),
        "fpr": round(float(fpr), 5) if not np.isnan(fpr) else np.nan,
        "fnr": round(float(fnr), 5) if not np.isnan(fnr) else np.nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "roc_auc": round(float(roc_auc_score(y_true, y_prob_pos)), 5) if has_both else np.nan,
        "pr_auc": round(float(average_precision_score(y_true, y_prob_pos)), 5) if has_both else np.nan,
    }


def compute_multiclass_metrics(
    y_true_enc: np.ndarray,
    y_pred_enc: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    y_true_enc = np.asarray(y_true_enc, dtype=np.int64)
    y_pred_enc = np.asarray(y_pred_enc, dtype=np.int64)
    n_cls = len(class_names)
    labels = list(range(n_cls))

    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=labels)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true_enc, y_pred_enc, labels=labels, zero_division=0)

    present = [i for i in labels if int(sup[i]) > 0]
    y_bin_mat = label_binarize(y_true_enc, classes=labels)

    roc_macro = np.nan
    pr_macro = np.nan
    if len(present) >= 2 and y_prob is not None and y_prob.shape[1] == n_cls:
        try:
            if len(present) == n_cls:
                roc_macro = float(roc_auc_score(y_true_enc, y_prob, multi_class="ovr", average="macro", labels=labels))
            else:
                roc_macro = float(np.mean([roc_auc_score(y_bin_mat[:, i], y_prob[:, i]) for i in present]))
        except Exception:
            pass
        try:
            pr_macro = float(np.mean([average_precision_score(y_bin_mat[:, i], y_prob[:, i]) for i in present]))
        except Exception:
            pass

    metrics = {
        "task_type": "multiclass",
        "auc_scheme": f"multiclass_{n_cls}class_OneVsRest_macro",
        "n_classes": n_cls,
        "n_samples": int(len(y_true_enc)),
        "accuracy": round(float(accuracy_score(y_true_enc, y_pred_enc)), 5),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true_enc, y_pred_enc)), 5),
        "f1_macro": round(float(f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)), 5),
        "f1_weighted": round(float(f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)), 5),
        "precision_macro": round(float(precision_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)), 5),
        "recall_macro": round(float(recall_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)), 5),
        "roc_auc_ovr_macro": round(roc_macro, 5) if not np.isnan(roc_macro) else np.nan,
        "pr_auc_ovr_macro": round(pr_macro, 5) if not np.isnan(pr_macro) else np.nan,
    }

    per_cls_df = pd.DataFrame({
        "class_name": class_names,
        "precision": np.round(prec, 5),
        "recall": np.round(rec, 5),
        "f1_score": np.round(f1, 5),
        "support": sup.astype(int),
    })
    return metrics, per_cls_df


_RSS_PROBE = r"""
import ctypes, gc, json, sys
import joblib, numpy as np, pandas as pd
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

m_path, p_path, s_path = sys.argv[1], sys.argv[2], sys.argv[3]
sample = joblib.load(s_path)
gc.collect()
b_rss, method = rss_mb()
_ = joblib.load(m_path)
gc.collect()
a_rss, _ = rss_mb()
pipe = joblib.load(p_path)
for i in range(min(30, len(sample))):
    pipe.predict(sample.iloc[i : i + 1])
gc.collect()
e_rss, _ = rss_mb()
print(json.dumps({
    "rss_method": method,
    "rss_baseline_mb": round(b_rss, 3) if b_rss is not None else None,
    "rss_model_delta_mb": round(max(0.0, a_rss - b_rss), 3) if (b_rss is not None and a_rss is not None) else None,
    "rss_after_infer_mb": round(e_rss, 3) if e_rss is not None else None,
}))
"""


def measure_edge_cost(
    pipeline: Pipeline,
    X_eval: pd.DataFrame,
    save_pipeline_path: Path,
    save_model_only_path: Path,
    train_time_sec: float,
    n_lat_samples: int = 200,
) -> dict:
    save_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_only_path.parent.mkdir(parents=True, exist_ok=True)

    prep = pipeline[:-1]
    clf = pipeline.named_steps["model"]
    if hasattr(clf, "n_jobs"):
        clf.set_params(n_jobs=1)

    joblib.dump(pipeline, save_pipeline_path)
    joblib.dump(clf, save_model_only_path)
    pipe_kb = save_pipeline_path.stat().st_size / 1024.0
    model_kb = save_model_only_path.stat().st_size / 1024.0

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
        sp = Path(tf.name)
    try:
        joblib.dump(X_eval.iloc[:30].copy(), sp)
        proc = subprocess.run(
            [sys.executable, "-c", _RSS_PROBE, str(save_model_only_path), str(save_pipeline_path), str(sp)],
            capture_output=True, text=True, timeout=60, check=False,
        )
        rss_info = json.loads(proc.stdout.strip().splitlines()[-1]) if (proc.returncode == 0 and proc.stdout.strip()) else {"rss_method": "error", "rss_model_delta_mb": np.nan, "rss_after_infer_mb": np.nan}
    finally:
        sp.unlink(missing_ok=True)

    # Batch preprocessing vs model-only süreleri
    t0 = time.perf_counter()
    X_tr = prep.transform(X_eval)
    prep_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = clf.predict(X_tr)
    mod_batch_sec = time.perf_counter() - t0

    # Tekil paket (batch=1) model-only ve end-to-end gecikmeler
    n_lat = min(n_lat_samples, len(X_eval))
    for i in range(min(15, n_lat)):
        clf.predict(X_tr[i : i + 1])

    m_lat = []
    for i in range(n_lat):
        ts = time.perf_counter()
        clf.predict(X_tr[i : i + 1])
        m_lat.append((time.perf_counter() - ts) * 1000.0)

    single_dfs = [X_eval.iloc[i : i + 1] for i in range(n_lat)]
    for i in range(min(10, n_lat)):
        pipeline.predict(single_dfs[i])

    e2e_lat = []
    for i in range(n_lat):
        ts = time.perf_counter()
        pipeline.predict(single_dfs[i])
        e2e_lat.append((time.perf_counter() - ts) * 1000.0)

    e2e_batch_sec = prep_sec + mod_batch_sec
    return {
        "train_time_sec": round(float(train_time_sec), 3),
        "model_only_size_kb": round(float(model_kb), 2),
        "pipeline_size_kb": round(float(pipe_kb), 2),
        "batch_preprocess_sec": round(float(prep_sec), 4),
        "batch_model_predict_sec": round(float(mod_batch_sec), 4),
        "throughput_model_only_pps": round(float(len(X_eval) / mod_batch_sec), 1) if mod_batch_sec > 0 else np.nan,
        "throughput_e2e_pps": round(float(len(X_eval) / e2e_batch_sec), 1) if e2e_batch_sec > 0 else np.nan,
        "model_only_latency_p50_ms": round(float(np.percentile(m_lat, 50)), 4),
        "model_only_latency_p95_ms": round(float(np.percentile(m_lat, 95)), 4),
        "e2e_latency_p50_ms": round(float(np.percentile(e2e_lat, 50)), 4),
        "e2e_latency_p95_ms": round(float(np.percentile(e2e_lat, 95)), 4),
        **rss_info,
    }