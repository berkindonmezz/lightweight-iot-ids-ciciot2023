import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABEL_COL, MODELS_DIR, RANDOM_STATE, RAW_DATA_DIR

MANIFEST_PATH = MODELS_DIR / "final_edge" / "frozen_manifest.json"
OUT_NPZ = Path(__file__).resolve().parent / "edge_benchmark_sample.npz"


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Muhurlu manifest bulunamadi: {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    features_top15 = manifest["models"]["top15_l1_balanced"]["features"]

    src_csv = RAW_DATA_DIR / "Merged09.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {src_csv}")

    print(f"Orneklem hazirlaniyor: {src_csv.name} ({len(features_top15)} ozellik)...")
    df = pd.read_csv(src_csv, usecols=features_top15 + [LABEL_COL], low_memory=False)
    for col in features_top15:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    df["y_bin"] = (df[LABEL_COL].astype(str).str.strip().str.upper() != "BENIGN").astype(np.int8)

    df_ben = df[df["y_bin"] == 0].sample(n=2000, random_state=RANDOM_STATE)
    df_atk = df[df["y_bin"] == 1].sample(n=8000, random_state=RANDOM_STATE)
    df_sub = (
        pd.concat([df_ben, df_atk], ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    X_raw = df_sub[features_top15].to_numpy(dtype=np.float32)
    y_sub = df_sub["y_bin"].to_numpy(dtype=np.int8)

    np.savez_compressed(
        OUT_NPZ,
        X_top15=X_raw,
        y_bin=y_sub,
        feature_names=np.array(features_top15),
    )
    print(f"Hazir -> {OUT_NPZ} ({len(X_raw):,} akis, {OUT_NPZ.stat().st_size / 1024.0:.1f} KB)")


if __name__ == "__main__":
    main()