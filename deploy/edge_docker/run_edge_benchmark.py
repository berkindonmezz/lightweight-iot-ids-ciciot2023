import csv
import ctypes
import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

APP_DIR = Path("/app")
MODELS_DIR = APP_DIR / "models_final_edge"
COMPILED_DIR = APP_DIR / "compiled_c"
SAMPLE_NPZ = APP_DIR / "edge_benchmark_sample.npz"

OUT_RESULTS_DIR = Path(os.environ.get("OUT_RESULTS_DIR", "/app/out_results"))
OUT_DOCS_DIR = Path(os.environ.get("OUT_DOCS_DIR", "/app/out_docs"))
PROFILE_TAG = os.environ.get("EDGE_PROFILE", "0.5cpu_256mb")

# Izole Linux surecinde ONNX Runtime Peak RSS (VmHWM), gecikme, throughput ve uyum olcumu
_ONNX_LINUX_PROBE = r"""
import json, sys, time
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)

def read_rss_mb():
    cur_mb = peak_mb = -1.0
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                cur_mb = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                peak_mb = int(line.split()[1]) / 1024.0
    return cur_mb, peak_mb

onnx_path, sample_npy, med_json, n_lat_s, out_probs_npy = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
)
raw_x = np.load(sample_npy).astype(np.float32)
medians = np.array(json.loads(med_json), dtype=np.float32)

def prep(arr):
    bad = ~np.isfinite(arr)
    if bad.any():
        arr = arr.copy()
        _, cols = np.where(bad)
        arr[bad] = medians[cols]
    return arr

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

n_s = min(n_lat_s, len(raw_x))
for i in range(min(50, n_s)):
    sess.run(None, {inp_name: prep(raw_x[i:i+1])})

lats = []
for i in range(n_s):
    t0 = time.perf_counter()
    x1 = prep(raw_x[i:i+1])
    sess.run(None, {inp_name: x1})
    lats.append((time.perf_counter() - t0) * 1e6)

x_all = prep(raw_x)
t0 = time.perf_counter()
out = sess.run(None, {inp_name: x_all})[1]
batch_sec = time.perf_counter() - t0

cur_mb, peak_mb = read_rss_mb()

if isinstance(out, np.ndarray):
    probs = out[:, 1].astype(np.float64)
else:
    probs = np.array([d.get(1, d.get("1", 0.0)) for d in out], dtype=np.float64)
np.save(out_probs_npy, probs)

print(json.dumps({
    "onnx_rss_cur_mb": round(cur_mb, 3),
    "onnx_rss_peak_mb": round(peak_mb, 3),
    "onnx_lat_p50_us": round(float(np.percentile(lats, 50)), 2),
    "onnx_lat_p95_us": round(float(np.percentile(lats, 95)), 2),
    "onnx_lat_p99_us": round(float(np.percentile(lats, 99)), 2),
    "onnx_throughput_pps": round(float(len(raw_x) / max(batch_sec, 1e-9)), 1),
}))
"""


def read_cgroup_limits() -> dict:
    info = {
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_limit_cores": "unlimited/unknown",
        "ram_limit_mb": "unlimited/unknown",
    }

    p_cpu_v2 = Path("/sys/fs/cgroup/cpu.max")
    if p_cpu_v2.exists():
        parts = p_cpu_v2.read_text().strip().split()
        if len(parts) == 2 and parts[0] != "max":
            info["cpu_limit_cores"] = round(int(parts[0]) / int(parts[1]), 3)
    else:
        q = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        p = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q.exists() and p.exists():
            qv, pv = int(q.read_text().strip()), int(p.read_text().strip())
            if qv > 0 and pv > 0:
                info["cpu_limit_cores"] = round(qv / pv, 3)

    p_mem_v2 = Path("/sys/fs/cgroup/memory.max")
    if p_mem_v2.exists():
        val = p_mem_v2.read_text().strip()
        if val != "max" and val.isdigit():
            info["ram_limit_mb"] = round(int(val) / (1024.0 ** 2), 1)
    else:
        p_mem_v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if p_mem_v1.exists():
            val = int(p_mem_v1.read_text().strip())
            if val < (1 << 60):
                info["ram_limit_mb"] = round(val / (1024.0 ** 2), 1)

    gcc_ver_file = COMPILED_DIR / "gcc_version.txt"
    if gcc_ver_file.exists():
        info["gcc_version"] = gcc_ver_file.read_text(encoding="utf-8").strip()
    else:
        info["gcc_version"] = "precompiled"
    return info


def run_linux_so_predict(so_path: Path, X_raw: np.ndarray) -> np.ndarray:
    x_c = np.ascontiguousarray(X_raw, dtype=np.float32)
    n_rows = int(x_c.shape[0])
    out = np.empty(n_rows, dtype=np.float64)

    lib = ctypes.CDLL(str(so_path))
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


def to_markdown_table(rows: list[dict], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def write_unified_report(out_results_dir: Path, out_docs_dir: Path, env_info: dict) -> None:
    csv_files = sorted(out_results_dir.glob("edge_benchmark_*.csv"))
    if not csv_files or not out_docs_dir.exists():
        return

    view_cols = [
        "profile", "cpu_limit", "ram_limit_mb", "model",
        "onnx_size_kb", "linux_bin_kb",
        "onnx_peak_ram_mb", "c_peak_ram_mb", "ram_ratio_onnx_vs_c_x",
        "onnx_p50_us", "c_p50_us", "c_p99_us", "speedup_c_vs_onnx_x",
        "c_pps", "c_vs_onnx_agree_%",
    ]

    all_rows = []
    for cf in csv_files:
        with open(cf, "r", encoding="utf-8", newline="") as f:
            all_rows.extend(list(csv.DictReader(f)))

    md_lines = [
        "# 07 — Kısıtlı Edge / Docker Dağıtım Benchmark Raporu\n",
        f"- **Konteyner Ortamı:** `{env_info['os']}` (`{env_info['arch']}`)",
        f"- **Derleme Stratejisi:** Multi-stage build (`{env_info['gcc_version']}`, `-O3`); çalışma imajında derleyici bulunmaz.",
        "- **Test Örneklemi:** `Merged09.csv` kilitli setinden 10.000 gerçek akış (2.000 Benign + 8.000 Saldırı)",
        "- **Bellek Ölçümü:** Linux çekirdeği `/proc/self/status` → `VmHWM` (Peak Resident Set Size)\n",
        "## Tüm Kısıtlı Profillerde ONNX Runtime vs. Saf C99 Kıyaslaması\n",
        to_markdown_table(all_rows, view_cols),
        "\n- Ham CSV çıktıları: `results/edge_benchmark/edge_benchmark_*.csv`",
    ]
    report_path = out_docs_dir / "07_edge_deployment_benchmark.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Birlesik rapor guncellendi -> docs/{report_path.name}")


def main():
    t_all = time.perf_counter()
    env_info = read_cgroup_limits()

    print("=============================================================================")
    print(f"KISITLI EDGE / DOCKER BENCHMARK ({PROFILE_TAG})")
    print("=============================================================================")
    print(f"OS / Arch        : {env_info['os']} ({env_info['arch']})")
    print(f"Cgroup CPU Limit : {env_info['cpu_limit_cores']} core")
    print(f"Cgroup RAM Limit : {env_info['ram_limit_mb']} MB")
    print(f"Build GCC        : {env_info['gcc_version']} (on-derlenmis ikililer)\n")

    manifest = json.loads((MODELS_DIR / "frozen_manifest.json").read_text(encoding="utf-8"))
    npz = np.load(SAMPLE_NPZ)
    X_top15 = npz["X_top15"].astype(np.float32)
    y_bin = npz["y_bin"].astype(np.int8)
    print(f"Test orneklemi   : {len(X_top15):,} gercek akis (Benign={(y_bin==0).sum():,}, Saldiri={(y_bin==1).sum():,})\n")

    rows = []
    for tag, mcfg in manifest["models"].items():
        n_feat = len(mcfg["features"])
        X_cur = np.ascontiguousarray(X_top15[:, :n_feat], dtype=np.float32)
        thr_01 = float(mcfg["thresholds_locked_from_m07"]["target_fpr_0.1%"])

        onnx_path = MODELS_DIR / mcfg["artifacts"]["onnx"]["file"]
        so_path = COMPILED_DIR / f"lib{tag}.so"
        bin_path = COMPILED_DIR / f"{tag}_bench_linux"

        so_kb = round(so_path.stat().st_size / 1024.0, 2)
        bin_kb = round(bin_path.stat().st_size / 1024.0, 2)
        onnx_kb = round(onnx_path.stat().st_size / 1024.0, 2)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf_bin:
            tmp_bin = Path(tf_bin.name)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf_npy:
            tmp_npy = Path(tf_npy.name)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf_probs:
            tmp_probs = Path(tf_probs.name)

        try:
            X_cur[:2000].tofile(tmp_bin)
            np.save(tmp_npy, X_cur)

            p_c = subprocess.run(
                [str(bin_path), str(tmp_bin), "2000", str(n_feat), "50"],
                capture_output=True, text=True, timeout=120, check=False,
            )
            if p_c.returncode != 0:
                raise RuntimeError(f"C benchmark hatasi ({tag}): {p_c.stderr}")
            c_bm = json.loads(p_c.stdout.strip().splitlines()[-1])

            p_ort = subprocess.run(
                [
                    sys.executable, "-c", _ONNX_LINUX_PROBE,
                    str(onnx_path), str(tmp_npy), json.dumps(mcfg["medians_float32"]), "500", str(tmp_probs),
                ],
                capture_output=True, text=True, timeout=180, check=False,
            )
            if p_ort.returncode != 0:
                raise RuntimeError(f"ONNX probe hatasi ({tag}): {p_ort.stderr}")
            ort_bm = json.loads(p_ort.stdout.strip().splitlines()[-1])
            prob_o = np.load(tmp_probs)
        finally:
            tmp_bin.unlink(missing_ok=True)
            tmp_npy.unlink(missing_ok=True)
            tmp_probs.unlink(missing_ok=True)

        prob_c = run_linux_so_predict(so_path, X_cur)
        agree_pct = round(float(((prob_c >= thr_01) == (prob_o >= thr_01)).mean() * 100.0), 5)

        ram_ratio = round(ort_bm["onnx_rss_peak_mb"] / max(c_bm["c_rss_peak_mb"], 0.01), 1)
        lat_ratio = round(ort_bm["onnx_lat_p50_us"] / max(c_bm["c_lat_p50_us"], 0.01), 1)

        row = {
            "profile": PROFILE_TAG,
            "cpu_limit": env_info["cpu_limit_cores"],
            "ram_limit_mb": env_info["ram_limit_mb"],
            "model": tag,
            "role": mcfg["role"],
            "n_features": n_feat,
            "onnx_size_kb": onnx_kb,
            "linux_so_kb": so_kb,
            "linux_bin_kb": bin_kb,
            "onnx_peak_ram_mb": ort_bm["onnx_rss_peak_mb"],
            "c_peak_ram_mb": c_bm["c_rss_peak_mb"],
            "ram_ratio_onnx_vs_c_x": ram_ratio,
            "onnx_p50_us": ort_bm["onnx_lat_p50_us"],
            "onnx_p95_us": ort_bm["onnx_lat_p95_us"],
            "onnx_p99_us": ort_bm["onnx_lat_p99_us"],
            "c_p50_us": c_bm["c_lat_p50_us"],
            "c_p95_us": c_bm["c_lat_p95_us"],
            "c_p99_us": c_bm["c_lat_p99_us"],
            "speedup_c_vs_onnx_x": lat_ratio,
            "onnx_pps": ort_bm["onnx_throughput_pps"],
            "c_pps": c_bm["c_throughput_pps"],
            "c_vs_onnx_agree_%": agree_pct,
        }
        rows.append(row)

        print(
            f">>> [{tag}] ({mcfg['role']}) <<<\n"
            f"  Boyut (ONNX / .so / ELF) : {onnx_kb:.1f} KB / {so_kb:.1f} KB / {bin_kb:.1f} KB\n"
            f"  Linux Peak RAM (VmHWM)   : ONNX={ort_bm['onnx_rss_peak_mb']:.2f} MB | Saf C99={c_bm['c_rss_peak_mb']:.2f} MB ({ram_ratio:.1f}x daha az)\n"
            f"  Tekil Gecikme P50/P95/P99: ONNX={ort_bm['onnx_lat_p50_us']:.1f}/{ort_bm['onnx_lat_p95_us']:.1f}/{ort_bm['onnx_lat_p99_us']:.1f} us | "
            f"Saf C99={c_bm['c_lat_p50_us']:.2f}/{c_bm['c_lat_p95_us']:.2f}/{c_bm['c_lat_p99_us']:.2f} us ({lat_ratio:.1f}x hizli)\n"
            f"  Throughput (pps)         : ONNX={ort_bm['onnx_throughput_pps']:,.0f} | Saf C99={c_bm['c_throughput_pps']:,.0f}\n"
            f"  Linux C = ONNX Uyumu     : %{agree_pct:.3f}\n"
        )

    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_RESULTS_DIR / f"edge_benchmark_{PROFILE_TAG}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta_path = OUT_RESULTS_DIR / f"edge_benchmark_{PROFILE_TAG}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "evaluated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "profile": PROFILE_TAG,
                "container_env": env_info,
                "n_sample_flows": int(len(X_top15)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_unified_report(OUT_RESULTS_DIR, OUT_DOCS_DIR, env_info)
    print(f"Benchmark tamamlandi ({time.perf_counter() - t_all:.1f} sn). Kayit -> results/edge_benchmark/{csv_path.name}")


if __name__ == "__main__":
    main()