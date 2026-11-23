import argparse
import ctypes
import gc
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
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import compute_binary_metrics, get_hardware_profile
from src.config import (
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

# Kilitli final test oncesi dondurulacak 3 model adayi
FROZEN_CANDIDATES = [
    {
        "tag": "top15_l0_baseline",
        "role": "Referans (Budanmamis)",
        "checkpoint": "Top15",
        "tier": "L0_Baseline",
        "features": FEATURES_TOP15,
        "params": {"n_estimators": 150, "num_leaves": 63, "max_depth": -1, "learning_rate": 0.05},
    },
    {
        "tag": "top15_l1_balanced",
        "role": "Ana Onerilen Model",
        "checkpoint": "Top15",
        "tier": "L1_Balanced",
        "features": FEATURES_TOP15,
        "params": {"n_estimators": 80, "num_leaves": 31, "max_depth": 7, "learning_rate": 0.07},
    },
    {
        "tag": "top10_l2_light",
        "role": "Ultra-Hafif Alternatif",
        "checkpoint": "Top10",
        "tier": "L2_Light",
        "features": FEATURES_TOP10,
        "params": {"n_estimators": 50, "num_leaves": 15, "max_depth": 6, "learning_rate": 0.08},
    },
]


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
            raise FileNotFoundError(f"Veri dosyasi bulunamadi: {fpath}")
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


def eval_tree_py(node: dict, x_row: np.ndarray) -> float:
    while "split_feature" in node:
        f_idx = int(node["split_feature"])
        thr = float(node["threshold"])
        if float(x_row[f_idx]) <= thr:
            node = node["left_child"]
        else:
            node = node["right_child"]
    return float(node["leaf_value"])


def emit_tree_c_body(node: dict, lines: list[str], indent: str = "    ") -> None:
    if "split_feature" not in node:
        lines.append(f"{indent}return {float(node['leaf_value']):.17g};")
        return

    dt = node.get("decision_type", "<=")
    if dt != "<=":
        raise ValueError(f"Desteklenmeyen decision_type: {dt}")

    f_idx = int(node["split_feature"])
    thr = float(node["threshold"])
    lines.append(f"{indent}if ((double)x[{f_idx}] <= {thr:.17g}) {{")
    emit_tree_c_body(node["left_child"], lines, indent + "    ")
    lines.append(f"{indent}}} else {{")
    emit_tree_c_body(node["right_child"], lines, indent + "    ")
    lines.append(f"{indent}}}")


def generate_c_model_source(
    clf: LGBMClassifier,
    medians: np.ndarray,
    thr_01: float,
    thr_02: float,
    thr_05: float,
    X_verify: np.ndarray,
    out_c_path: Path,
) -> dict:
    """LightGBM agaclarini sifir bagimlilikli saf C99 kaynak koduna cevirir."""
    dump = clf.booster_.dump_model()
    trees = dump["tree_info"]
    n_features = len(medians)

    # LightGBM raw_score ile bias kontrolu (surum farklarina karsi %100 guvence)
    raw_lgb = clf.predict(X_verify[:10], raw_score=True)
    raw_py = np.array(
        [sum(eval_tree_py(t["tree_structure"], X_verify[i]) for t in trees) for i in range(len(raw_lgb))],
        dtype=np.float64,
    )
    bias = float(np.mean(raw_lgb - raw_py))
    if abs(bias) < 1e-12:
        bias = 0.0
    max_verify_err = float(np.max(np.abs((raw_py + bias) - raw_lgb)))
    if max_verify_err > 1e-5:
        raise RuntimeError(f"Agac dokumu LightGBM raw_score ile uyusmuyor! max_err={max_verify_err}")

    total_leaves = 0
    lines = [
        "/* Otomatik uretilmis sifir-bagimlilikli saf C99 Lightweight IoT IDS modeli */",
        "#include <math.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "#ifdef _WIN32",
        "#define EXPORT_API __declspec(dllexport)",
        "#else",
        "#define EXPORT_API",
        "#endif",
        "",
        f"#define IDS_N_FEATURES {n_features}",
        f"#define IDS_N_TREES {len(trees)}",
        f"#define IDS_THR_FPR_01 {thr_01:.17g}",
        f"#define IDS_THR_FPR_02 {thr_02:.17g}",
        f"#define IDS_THR_FPR_05 {thr_05:.17g}",
        f"static const double IDS_INIT_BIAS = {bias:.17g};",
        "static const float IDS_MEDIANS[IDS_N_FEATURES] = {",
        "    " + ", ".join(f"{float(m):.9e}f" for m in medians),
        "};",
        "",
    ]

    for idx, t in enumerate(trees):
        total_leaves += int(t.get("num_leaves", 0))
        lines.append(f"static double tree_{idx}(const float *x) {{")
        emit_tree_c_body(t["tree_structure"], lines, "    ")
        lines.append("}")
        lines.append("")

    # Tekil akis E2E olasilik tahmini (Inf/NaN medyan doldurma + agac toplami + sigmoid)
    lines.extend([
        "EXPORT_API double ids_predict_prob(const float *raw_x) {",
        "    float x[IDS_N_FEATURES];",
        "    for (int i = 0; i < IDS_N_FEATURES; ++i) {",
        "        float v = raw_x[i];",
        "        x[i] = isfinite(v) ? v : IDS_MEDIANS[i];",
        "    }",
        "    double s = IDS_INIT_BIAS;",
    ])
    for idx in range(len(trees)):
        lines.append(f"    s += tree_{idx}(x);")
    lines.extend([
        "    return 1.0 / (1.0 + exp(-s));",
        "}",
        "",
        "EXPORT_API int ids_predict_label(const float *raw_x, double threshold) {",
        "    return ids_predict_prob(raw_x) >= threshold ? 1 : 0;",
        "}",
        "",
        "EXPORT_API void ids_predict_batch(const float *raw_x, int n_rows, double *out_probs) {",
        "    for (int r = 0; r < n_rows; ++r) {",
        "        out_probs[r] = ids_predict_prob(raw_x + ((size_t)r * IDS_N_FEATURES));",
        "    }",
        "}",
        "",
    ])

    out_c_path.parent.mkdir(parents=True, exist_ok=True)
    out_c_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "n_trees": len(trees),
        "total_leaves": total_leaves,
        "bias": bias,
        "c_source_kb": round(out_c_path.stat().st_size / 1024.0, 2),
    }


def write_bench_driver_c(out_driver_path: Path, n_features: int) -> None:
    """Saf C ikilisinin gercek OS RSS (MB), tekil gecikme (us) ve throughput (pps) olcucu surucusu."""
    code = f"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
static double now_us(void) {{
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) {{
        QueryPerformanceFrequency(&freq);
        init = 1;
    }}
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e6 / (double)freq.QuadPart;
}}
static void get_rss_mb(double *cur_mb, double *peak_mb) {{
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {{
        *cur_mb = (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
        *peak_mb = (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    }} else {{
        *cur_mb = -1.0;
        *peak_mb = -1.0;
    }}
}}
#else
#include <time.h>
#include <sys/resource.h>
static double now_us(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}}
static void get_rss_mb(double *cur_mb, double *peak_mb) {{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {{
        *peak_mb = (double)ru.ru_maxrss / 1024.0;
        *cur_mb = *peak_mb;
    }} else {{
        *cur_mb = -1.0;
        *peak_mb = -1.0;
    }}
}}
#endif

extern double ids_predict_prob(const float *raw_x);

static int cmp_double(const void *a, const void *b) {{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}}

int main(int argc, char **argv) {{
    if (argc < 3) {{
        fprintf(stderr, "Kullanim: bench.exe <sample_bin> <n_samples> [batch_loops]\\n");
        return 1;
    }}
    const char *bin_path = argv[1];
    int n_samples = atoi(argv[2]);
    int loops = (argc >= 4) ? atoi(argv[3]) : 50;
    const int n_feat = {n_features};

    FILE *fp = fopen(bin_path, "rb");
    if (!fp) {{
        fprintf(stderr, "Dosya acilamadi: %s\\n", bin_path);
        return 2;
    }}
    float *data = (float *)malloc((size_t)n_samples * n_feat * sizeof(float));
    double *lats = (double *)malloc((size_t)n_samples * sizeof(double));
    if (!data || !lats) {{
        fclose(fp);
        return 3;
    }}
    size_t got = fread(data, sizeof(float), (size_t)n_samples * n_feat, fp);
    fclose(fp);
    if ((int)got < n_samples * n_feat) {{
        n_samples = (int)(got / n_feat);
    }}

    /* Warm-up */
    volatile double sink = 0.0;
    int warmup = n_samples < 200 ? n_samples : 200;
    for (int i = 0; i < warmup; ++i) {{
        sink += ids_predict_prob(data + (size_t)i * n_feat);
    }}

    /* Tekil akis (batch=1) gecikme olcumu (mikrosaniye) */
    for (int i = 0; i < n_samples; ++i) {{
        double t0 = now_us();
        double p = ids_predict_prob(data + (size_t)i * n_feat);
        double t1 = now_us();
        lats[i] = t1 - t0;
        sink += p;
    }}

    qsort(lats, (size_t)n_samples, sizeof(double), cmp_double);
    double p50_us = lats[(int)(0.50 * (n_samples - 1))];
    double p95_us = lats[(int)(0.95 * (n_samples - 1))];
    double p99_us = lats[(int)(0.99 * (n_samples - 1))];

    /* Tek cekirdek throughput olcumu */
    double tb0 = now_us();
    long long total_inf = (long long)loops * n_samples;
    for (int l = 0; l < loops; ++l) {{
        for (int i = 0; i < n_samples; ++i) {{
            sink += ids_predict_prob(data + (size_t)i * n_feat);
        }}
    }}
    double tb1 = now_us();
    double elapsed_sec = (tb1 - tb0) / 1e6;
    double pps = elapsed_sec > 0.0 ? ((double)total_inf / elapsed_sec) : 0.0;

    double cur_mb = 0.0, peak_mb = 0.0;
    get_rss_mb(&cur_mb, &peak_mb);

    printf("{{\\"c_rss_cur_mb\\": %.3f, \\"c_rss_peak_mb\\": %.3f, \\"c_lat_p50_us\\": %.3f, \\"c_lat_p95_us\\": %.3f, \\"c_lat_p99_us\\": %.3f, \\"c_throughput_pps\\": %.1f, \\"checksum\\": %.6f}}\\n",
           cur_mb, peak_mb, p50_us, p95_us, p99_us, pps, sink);

    free(data);
    free(lats);
    return 0;
}}
"""
    out_driver_path.write_text(code.strip() + "\n", encoding="utf-8")


def compile_c_targets(c_model_path: Path, c_driver_path: Path, dll_path: Path, exe_path: Path) -> None:
    gcc_exe = shutil.which("gcc")
    if not gcc_exe:
        raise RuntimeError("gcc bulunamadi! PATH ayarini kontrol edin.")

    # 1) Paylasimli kutuphane (.dll / .so)
    cmd_dll = [gcc_exe, "-O3", "-shared", "-static-libgcc", str(c_model_path), "-o", str(dll_path), "-lm"]
    r1 = subprocess.run(cmd_dll, capture_output=True, text=True, check=False)
    if r1.returncode != 0:
        raise RuntimeError(f"DLL derleme hatasi:\n{r1.stderr}")

    # 2) Bagimsiz benchmark ikilisi (.exe)
    cmd_exe = [gcc_exe, "-O3", "-static-libgcc", str(c_model_path), str(c_driver_path), "-o", str(exe_path), "-lm"]
    if sys.platform == "win32":
        cmd_exe.append("-lpsapi")
    r2 = subprocess.run(cmd_exe, capture_output=True, text=True, check=False)
    if r2.returncode != 0:
        raise RuntimeError(f"EXE derleme hatasi:\n{r2.stderr}")


def run_dll_batch_predict(dll_path: Path, X_raw_f32: np.ndarray) -> np.ndarray:
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

    lib.ids_predict_batch(
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_rows,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out


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


def check_against_exp5_onnx(tag: str, X_ev_imp: np.ndarray, prob_c: np.ndarray) -> tuple[float | None, float | None]:
    """Exp 5'te kaydedilmis .onnx dosyasi varsa birebir olasilik uyumunu dogrular."""
    onnx_path = MODELS_DIR / "exp5_lightweighting" / f"{tag}.onnx"
    if not onnx_path.exists():
        return None, None
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: X_ev_imp.astype(np.float32)})[1]
    if isinstance(out, np.ndarray):
        prob_ort = out[:, 1].astype(np.float64)
    else:
        prob_ort = np.array([d.get(1, d.get("1", 0.0)) for d in out], dtype=np.float64)
    diff = np.abs(prob_c - prob_ort)
    return round(float(np.max(diff)), 7), round(float(np.mean(diff)), 7)


def main():
    parser = argparse.ArgumentParser(
        description="Exp 5b — Dondurulacak 3 Modelin Saf C99 Koduna Donusturulmesi ve Native Benchmark"
    )
    parser.add_argument("--quick", action="store_true", help="Hizli dogrulama modu (~20 sn).")
    args = parser.parse_args()

    t_all = time.perf_counter()
    hw = get_hardware_profile()

    gcc_ver_proc = subprocess.run(["gcc", "--version"], capture_output=True, text=True, check=False)
    gcc_version_line = gcc_ver_proc.stdout.splitlines()[0].strip() if gcc_ver_proc.returncode == 0 else "unknown"

    assert CAL_FILE not in TRAIN_FILES and EVAL_FILE not in TRAIN_FILES
    assert CAL_FILE not in TEST_FILES_LOCKED and EVAL_FILE not in TEST_FILES_LOCKED

    if args.quick:
        exp_dir = RESULTS_DIR / "exp5b_c_native_quick"
        mod_dir = MODELS_DIR / "exp5b_c_native_quick"
        tr_files = TRAIN_FILES[:2]
        tr_ben, tr_atk, ev_frac = 0.20, 0.03, 0.15
        bench_n, bench_loops = 1000, 20
    else:
        exp_dir = RESULTS_DIR / "exp5b_c_native"
        mod_dir = MODELS_DIR / "exp5b_c_native"
        tr_files = TRAIN_FILES
        tr_ben, tr_atk, ev_frac = 1.00, 0.25, 1.00
        bench_n, bench_loops = 2000, 50  # 100,000 tek cekirdek cikarim

    exp_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================================")
    print(f"EXP 5B — SAF C99 NATIVE EXPORT & EDGE BENCHMARK ({'QUICK' if args.quick else 'TAM'})")
    print("=============================================================================")
    print(f"Derleyici : {gcc_version_line}")
    print(f"Donanim   : {hw['os']} | CPU={hw['logical_cpu_count']} | RAM={hw['total_ram_gb']} GB")
    print(f"Adaylar   : {[c['tag'] for c in FROZEN_CANDIDATES]}")
    print(f"Test Seti : {TEST_FILES_LOCKED} (KILITLI — DOKUNULMUYOR)\n")

    print("Veriler yukleniyor...")
    df_tr = load_files_ram_safe(tr_files, FEATURES_TOP15, benign_frac=tr_ben, attack_frac=tr_atk)
    df_cal = load_files_ram_safe([CAL_FILE], FEATURES_TOP15, benign_frac=ev_frac, attack_frac=ev_frac)
    df_eval = load_files_ram_safe([EVAL_FILE], FEATURES_TOP15, benign_frac=ev_frac, attack_frac=ev_frac)

    y_tr = df_tr["y_bin"].values
    y_cal = df_cal["y_bin"].values
    y_ev = df_eval["y_bin"].values
    print(f"Train: {len(df_tr):,} | Cal ({CAL_FILE}): {len(df_cal):,} | Eval ({EVAL_FILE}): {len(df_eval):,}\n")

    # Exp 5 sonuclarini (varsa) yan yana kiyaslama tablosu icin yukle
    exp5_csv = RESULTS_DIR / "exp5_lightweighting" / "exp5_pruning_and_onnx_results.csv"
    exp5_df = pd.read_csv(exp5_csv) if (not args.quick and exp5_csv.exists()) else None

    rows = []
    compare_rows = []

    for cand in FROZEN_CANDIDATES:
        tag = cand["tag"]
        cols = cand["features"]
        p = cand["params"]

        print(f">>> [{tag}] ({cand['role']} | {len(cols)} ozellik | {p['n_estimators']} agac, {p['num_leaves']} yaprak) <<<")

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(df_tr[cols]).astype(np.float32)
        medians = imputer.statistics_.astype(np.float32)

        X_cal_imp = preprocess_numpy_fast(df_cal[cols].to_numpy(dtype=np.float32), medians)
        X_ev_raw = df_eval[cols].to_numpy(dtype=np.float32)
        X_ev_imp = preprocess_numpy_fast(X_ev_raw, medians)

        clf = LGBMClassifier(
            n_estimators=p["n_estimators"],
            num_leaves=p["num_leaves"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr_imp, y_tr)

        prob_cal_lgb = clf.predict_proba(X_cal_imp)[:, 1]
        prob_ev_lgb = clf.predict_proba(X_ev_imp)[:, 1]

        thr_05 = threshold_for_target_fpr(y_cal, prob_cal_lgb, 0.005)
        thr_02 = threshold_for_target_fpr(y_cal, prob_cal_lgb, 0.002)
        thr_01 = threshold_for_target_fpr(y_cal, prob_cal_lgb, 0.001)

        # Saf C99 kaynak kodu uret ve derle
        c_model_path = mod_dir / f"{tag}_model.c"
        c_driver_path = mod_dir / f"{tag}_bench_main.c"
        dll_path = mod_dir / (f"{tag}.dll" if sys.platform == "win32" else f"lib{tag}.so")
        exe_path = mod_dir / (f"{tag}_bench.exe" if sys.platform == "win32" else f"{tag}_bench")

        t_gen = time.perf_counter()
        c_meta = generate_c_model_source(clf, medians, thr_01, thr_02, thr_05, X_tr_imp, c_model_path)
        write_bench_driver_c(c_driver_path, len(cols))
        compile_c_targets(c_model_path, c_driver_path, dll_path, exe_path)
        compile_sec = time.perf_counter() - t_gen

        dll_kb = round(dll_path.stat().st_size / 1024.0, 2)
        exe_kb = round(exe_path.stat().st_size / 1024.0, 2)

        # 1) Tum Merged08 (712,220 satir) uzerinde derlenmis C DLL ile tahmin & esdegerlik testi
        prob_ev_c = run_dll_batch_predict(dll_path, X_ev_raw)
        max_diff_lgb = float(np.max(np.abs(prob_ev_c - prob_ev_lgb)))
        mean_diff_lgb = float(np.mean(np.abs(prob_ev_c - prob_ev_lgb)))

        pred_01_c = (prob_ev_c >= thr_01).astype(np.int8)
        pred_01_lgb = (prob_ev_lgb >= thr_01).astype(np.int8)
        agree_01_pct = float((pred_01_c == pred_01_lgb).mean() * 100.0)

        m01 = compute_binary_metrics(y_ev, pred_01_c, prob_ev_c)
        l99_p01, l99_f01 = compute_99pct_normal_metrics(y_ev, pred_01_c)

        # Exp 5 ONNX ile capraz esdegerlik (yalnizca tam modda)
        max_diff_ort, mean_diff_ort = (None, None)
        if not args.quick:
            max_diff_ort, mean_diff_ort = check_against_exp5_onnx(tag, X_ev_imp, prob_ev_c)

        # 2) Python'siz saf C EXE ile gercek OS RSS (MB) ve mikrosaniye gecikme olcumu
        n_b = min(bench_n, len(X_ev_raw))
        native_bm = run_native_exe_benchmark(exe_path, X_ev_raw[:n_b], loops=bench_loops)

        c_lat_p50_ms = round(native_bm["c_lat_p50_us"] / 1000.0, 5)
        c_lat_p95_ms = round(native_bm["c_lat_p95_us"] / 1000.0, 5)
        c_lat_p99_ms = round(native_bm["c_lat_p99_us"] / 1000.0, 5)

        # Dagitim JSON kaydi (final test ve GUI icin hazir)
        cfg_out = {
            "tag": tag,
            "role": cand["role"],
            "checkpoint": cand["checkpoint"],
            "tier": cand["tier"],
            "features": cols,
            "medians_float32": [float(v) for v in medians],
            "hyperparams": p,
            "thresholds_cal_m07": {
                "target_fpr_0.5%": round(thr_05, 6),
                "target_fpr_0.2%": round(thr_02, 6),
                "target_fpr_0.1%": round(thr_01, 6),
            },
            "c_source_file": c_model_path.name,
            "dll_file": dll_path.name,
            "bench_exe_file": exe_path.name,
        }
        (mod_dir / f"{tag}_c_config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")

        row = {
            "tag": tag,
            "role": cand["role"],
            "n_features": len(cols),
            "n_trees": c_meta["n_trees"],
            "total_leaves": c_meta["total_leaves"],
            "compile_sec": round(compile_sec, 2),
            "c_source_kb": c_meta["c_source_kb"],
            "dll_size_kb": dll_kb,
            "exe_size_kb": exe_kb,
            "roc_auc": m01["roc_auc"],
            "fpr01_real_%": round(m01["fpr"] * 100.0, 3),
            "fpr01_rec_%": round(m01["attack_recall"] * 100.0, 3),
            "fpr01_live99_p": l99_p01,
            "fpr01_live99_f1": l99_f01,
            "max_prob_diff_vs_lgb": round(max_diff_lgb, 8),
            "mean_prob_diff_vs_lgb": round(mean_diff_lgb, 8),
            "pred_agree_vs_lgb_%": round(agree_01_pct, 5),
            "max_prob_diff_vs_exp5_onnx": max_diff_ort,
            "c_rss_cur_mb": native_bm["c_rss_cur_mb"],
            "c_rss_peak_mb": native_bm["c_rss_peak_mb"],
            "c_lat_p50_us": native_bm["c_lat_p50_us"],
            "c_lat_p95_us": native_bm["c_lat_p95_us"],
            "c_lat_p99_us": native_bm["c_lat_p99_us"],
            "c_lat_p50_ms": c_lat_p50_ms,
            "c_throughput_pps": native_bm["c_throughput_pps"],
        }
        rows.append(row)

        print(
            f"  Derleme={compile_sec:.1f}sn | Boyut(C/DLL/EXE)={c_meta['c_source_kb']:.1f}/{dll_kb:.1f}/{exe_kb:.1f}KB\n"
            f"  @0.1%: FPR={row['fpr01_real_%']:.3f}% Rec={row['fpr01_rec_%']:.2f}% Live99P={l99_p01:.3f} | "
            f"Agree(LGB)={agree_01_pct:.3f}% (maxDiff={max_diff_lgb:.2e}, ortDiff={max_diff_ort})\n"
            f"  Saf C Native: RAM(Cur/Peak)={native_bm['c_rss_cur_mb']:.2f}/{native_bm['c_rss_peak_mb']:.2f}MB | "
            f"Lat P50/P95/P99={native_bm['c_lat_p50_us']:.2f}/{native_bm['c_lat_p95_us']:.2f}/{native_bm['c_lat_p99_us']:.2f} us | "
            f"Hiz={native_bm['c_throughput_pps']:,.0f} pps\n"
        )

        # 3'lu kiyaslama satiri (Sklearn vs ONNX vs Saf C)
        if exp5_df is not None:
            m_e5 = exp5_df[(exp5_df["checkpoint"] == cand["checkpoint"]) & (exp5_df["tier"] == cand["tier"])]
            if not m_e5.empty:
                r5 = m_e5.iloc[0]
                sk_ms = float(r5["sk_e2e_lat_p50_ms"])
                ort_ms = float(r5["onnx_e2e_lat_p50_ms"])
                c_ms = max(native_bm["c_lat_p50_us"] / 1000.0, 1e-6)
                compare_rows.append({
                    "model": tag,
                    "role": cand["role"],
                    "sklearn_ram_mb": r5["sklearn_rss_total_after_infer_mb"],
                    "onnx_ram_mb": r5["onnx_rss_total_after_infer_mb"],
                    "c_native_peak_ram_mb": native_bm["c_rss_peak_mb"],
                    "ram_reduction_vs_sklearn_x": round(float(r5["sklearn_rss_total_after_infer_mb"]) / max(native_bm["c_rss_peak_mb"], 0.01), 1),
                    "sklearn_p50_us": round(sk_ms * 1000.0, 1),
                    "onnx_p50_us": round(ort_ms * 1000.0, 1),
                    "c_native_p50_us": native_bm["c_lat_p50_us"],
                    "speedup_c_vs_sklearn_x": round(sk_ms / c_ms, 1),
                    "speedup_c_vs_onnx_x": round(ort_ms / c_ms, 1),
                    "onnx_size_kb": r5["onnx_size_kb"],
                    "c_dll_size_kb": dll_kb,
                    "c_throughput_pps": native_bm["c_throughput_pps"],
                })

    res_df = pd.DataFrame(rows)
    res_df.to_csv(exp_dir / "exp5b_c_native_results.csv", index=False)

    comp_df = pd.DataFrame(compare_rows)
    if not comp_df.empty:
        comp_df.to_csv(exp_dir / "exp5b_runtime_comparison_sklearn_onnx_c.csv", index=False)
        print("=== 3'LU CALISMA ZAMANI KIYASLAMASI: SKLEARN vs ONNX vs SAF C99 (TEK CEKIRDEK) ===")
        print(comp_df.to_string(index=False))

    with open(exp_dir / "exp5b_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "is_quick": bool(args.quick),
                "gcc_version": gcc_version_line,
                "candidates": [c["tag"] for c in FROZEN_CANDIDATES],
                "hardware": hw,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not args.quick:
        md_lines = [
            "# Exp 5b — Saf C99 Native Dönüşümü ve 3'lü Çalışma Zamanı Kıyaslaması\n",
            f"- **Derleyici:** `{gcc_version_line}` (`-O3 -static-libgcc`)",
            f"- **Değerlendirme:** `{EVAL_FILE}` (712.220 satır) | **Kilitli Test:** `{TEST_FILES_LOCKED}` (henüz açılmadı)\n",
            "## 1. Saf C99 Native Sonuçları ve Eşdeğerlik Doğrulaması\n",
            res_df.to_markdown(index=False),
        ]
        if not comp_df.empty:
            md_lines.append("\n## 2. Sklearn vs. ONNX Runtime vs. Saf C99 (RAM, Gecikme ve Hızlanma)\n")
            md_lines.append(comp_df.to_markdown(index=False))

        notes_path = DOCS_DIR / "05b_exp5b_c_native_notes.md"
        notes_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"\nRapor notlari -> {notes_path}")

    print(f"Exp 5b tamamlandi ({time.perf_counter() - t_all:.1f} sn). Ciktilar -> {exp_dir}")


if __name__ == "__main__":
    main()