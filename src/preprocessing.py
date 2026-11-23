import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURES_33,
    LABEL_COL,
    RAW_DATA_DIR,
    map_label_to_8_classes,
)


def load_csv_files(
    filenames: list[str],
    feature_cols: list[str] = FEATURES_33,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Belirtilen MergedXX.csv dosyalarını yükler, Inf/-Inf değerlerini NaN'a çevirir,
    binary (y_bin) ve 8-sınıf (y_cat8) etiketlerini ekler ve (istenirse)
    sınıf bazlı orantılı alt-örnekleme uygular.
    """
    usecols = list(dict.fromkeys(feature_cols + [LABEL_COL]))
    frames = []

    for fname in filenames:
        fpath = RAW_DATA_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {fpath}")
        df_part = pd.read_csv(fpath, usecols=usecols, low_memory=False)
        df_part["_source_file"] = fname
        frames.append(df_part)

    df = pd.concat(frames, ignore_index=True)

    # Sayısal sütunlarda Inf / -Inf temizliği (imputer tarafından doldurulmak üzere NaN yapılır)
    for col in feature_cols:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    raw_label = df[LABEL_COL].astype(str).str.strip()
    df["y_bin"] = (raw_label.str.upper() != "BENIGN").astype(np.int8)
    df["y_cat8"] = raw_label.map(map_label_to_8_classes)

    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        parts = []
        for _, grp in df.groupby("y_cat8", sort=False):
            n_take = max(1, int(round(len(grp) * sample_frac)))
            n_take = min(n_take, len(grp))
            parts.append(grp.sample(n=n_take, random_state=random_state))
        df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return df


def build_feature_pipeline(estimator, scale_numeric: bool = False) -> Pipeline:
    """
    Yalnızca X_train üzerinde fit edilecek sızıntısız Pipeline üretir.
    Ağaç modelleri için scale_numeric=False, Logistic Regression için True kullanılır.
    """
    steps = [("preprocessor", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)