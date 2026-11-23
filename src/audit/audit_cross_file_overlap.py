import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_8_ORDER,
    FEATURES_33,
    LABEL_COL,
    RAW_DATA_DIR,
    RESULTS_DIR,
    TRAIN_FILES,
    map_label_to_8_classes,
)

EVAL_FILES = ["Merged07.csv", "Merged08.csv"]

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


def read_clean_chunk(fname: str) -> pd.DataFrame:
    fpath = RAW_DATA_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {fpath}")
    usecols = FEATURES_24 + [LABEL_COL]
    df = pd.read_csv(fpath, usecols=usecols, low_memory=False)
    for col in FEATURES_24:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def hash_features(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df[FEATURES_24], index=False).values


def main():
    t0 = time.perf_counter()
    out_dir = RESULTS_DIR / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================================")
    print("ÖN DENETİM — 24 ÖZELLİK ÜZERİNDE DOSYALAR ARASI BİREBİR ÖRTÜŞME KONTROLÜ")
    print("=============================================================================")
    print(f"Referans Train dosyaları (TAM %100): {TRAIN_FILES}")
    print(f"Kontrol edilen dosyalar            : {EVAL_FILES}\n")

    # 1) Merged01-04'ün tamamının hash havuzunu RAM dostu çıkar
    tr_all_hashes = []
    tr_ben_hashes = []
    tr_atk_hashes = []
    total_tr_rows = 0

    for fname in TRAIN_FILES:
        df = read_clean_chunk(fname)
        h = hash_features(df)
        is_atk = (df[LABEL_COL].astype(str).str.strip().str.upper() != "BENIGN").values

        tr_all_hashes.append(np.unique(h))
        tr_ben_hashes.append(np.unique(h[~is_atk]))
        tr_atk_hashes.append(np.unique(h[is_atk]))
        total_tr_rows += len(df)
        print(f"  Tarandı: {fname} ({len(df):,} satır)")

        del df, h, is_atk
        gc.collect()

    train_unique = np.unique(np.concatenate(tr_all_hashes))
    train_ben_unique = np.unique(np.concatenate(tr_ben_hashes))
    train_atk_unique = np.unique(np.concatenate(tr_atk_hashes))

    print(
        f"\nTrain Havuzu ({total_tr_rows:,} satır) -> "
        f"Tekil vektör: {len(train_unique):,} "
        f"(Benign tekil: {len(train_ben_unique):,}, Attack tekil: {len(train_atk_unique):,})\n"
    )

    # 2) Merged07 ve Merged08'i sınıf bazlı denetle
    all_rows = []
    for ef in EVAL_FILES:
        df_ev = read_clean_chunk(ef)
        h_ev = hash_features(df_ev)
        raw_lbl = df_ev[LABEL_COL].astype(str).str.strip()
        cat8 = raw_lbl.map(map_label_to_8_classes).values
        is_atk = (raw_lbl.str.upper() != "BENIGN").values

        in_train = np.isin(h_ev, train_unique)
        # Çapraz etiket çelişkisi (Train'de Benign olup Eval'de Attack olan veya tersi)
        ben_in_tr_atk = int((~is_atk & np.isin(h_ev, train_atk_unique)).sum())
        atk_in_tr_ben = int((is_atk & np.isin(h_ev, train_ben_unique)).sum())

        print(f"--- {ef} ({len(df_ev):,} satır) | Çapraz Etiket Çelişkisi: Ben->Atk={ben_in_tr_atk}, Atk->Ben={atk_in_tr_ben} ---")
        file_rows = []
        for cat in CLASS_8_ORDER:
            cmask = cat8 == cat
            tot = int(cmask.sum())
            ov = int((cmask & in_train).sum())
            row = {
                "eval_file": ef,
                "category": cat,
                "eval_rows": tot,
                "exact_overlap_with_train": ov,
                "overlap_pct": round(ov / tot * 100.0, 2) if tot > 0 else 0.0,
                "strictly_unseen_rows": tot - ov,
            }
            file_rows.append(row)
            all_rows.append(row)

        tot_f = len(df_ev)
        ov_f = int(in_train.sum())
        tot_row = {
            "eval_file": ef,
            "category": "TOPLAM",
            "eval_rows": tot_f,
            "exact_overlap_with_train": ov_f,
            "overlap_pct": round(ov_f / tot_f * 100.0, 2),
            "strictly_unseen_rows": tot_f - ov_f,
        }
        file_rows.append(tot_row)
        all_rows.append(tot_row)

        print(pd.DataFrame(file_rows).drop(columns=["eval_file"]).to_string(index=False))
        print()

        del df_ev, h_ev, cat8
        gc.collect()

    out_csv = out_dir / "cross_file_overlap_audit.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Denetim tamamlandı ({time.perf_counter() - t0:.1f} sn). Kayıt -> {out_csv}")


if __name__ == "__main__":
    main()