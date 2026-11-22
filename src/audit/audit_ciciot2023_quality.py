import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "raw_ciciot2023"


def main():
    t0 = time.perf_counter()
    f_train = DATA_DIR / "Merged01.csv"
    f_val = DATA_DIR / "Merged07.csv"

    print(f"Yükleniyor: {f_train.name} (Train adayı) ve {f_val.name} (Val adayı)...")
    df_tr = pd.read_csv(f_train)
    df_va = pd.read_csv(f_val)

    label_col = "Label" if "Label" in df_tr.columns else df_tr.columns[-1]
    feat_cols = [c for c in df_tr.columns if c != label_col]

    print(f"\n=== 1. TEMEL YAPI ===")
    print(f"Merged01: {df_tr.shape[0]:,} satır x {df_tr.shape[1]} sütun")
    print(f"Merged07: {df_va.shape[0]:,} satır x {df_va.shape[1]} sütun")
    print(f"Sütunlar birebir aynı mı?: {list(df_tr.columns) == list(df_va.columns)}")
    print(f"Özellik sayısı: {len(feat_cols)} | Etiket sütunu: '{label_col}'")

    print(f"\n=== 2. TÜM ÖZELLİK SÜTUNLARI VE TİPLERİ ({len(feat_cols)} özellik) ===")
    dtype_df = pd.DataFrame({
        "sutun": feat_cols,
        "dtype": [str(df_tr[c].dtype) for c in feat_cols],
        "tr_nan": [int(df_tr[c].isna().sum()) for c in feat_cols],
        "va_nan": [int(df_va[c].isna().sum()) for c in feat_cols],
        "tr_inf": [
            int(np.isinf(df_tr[c]).sum()) if np.issubdtype(df_tr[c].dtype, np.number) else 0
            for c in feat_cols
        ],
        "va_inf": [
            int(np.isinf(df_va[c]).sum()) if np.issubdtype(df_va[c].dtype, np.number) else 0
            for c in feat_cols
        ],
        "tr_nunique": [int(df_tr[c].nunique(dropna=True)) for c in feat_cols],
    })
    print(dtype_df.to_string(index=False))

    zero_var = dtype_df[dtype_df["tr_nunique"] <= 1]["sutun"].tolist()
    non_num = dtype_df[~dtype_df["dtype"].str.contains("int|float")]["sutun"].tolist()
    has_nan_or_inf = dtype_df[
        (dtype_df["tr_nan"] > 0)
        | (dtype_df["va_nan"] > 0)
        | (dtype_df["tr_inf"] > 0)
        | (dtype_df["va_inf"] > 0)
    ]["sutun"].tolist()

    print(f"\nSıfır varyans (sabit) sütunlar : {zero_var if zero_var else 'Yok'}")
    print(f"Sayısal olmayan özellikler     : {non_num if non_num else 'Yok'}")
    print(f"NaN veya Inf içeren sütunlar   : {has_nan_or_inf if has_nan_or_inf else 'Yok'}")

    # 3. Hızlı hash tabanlı duplicate ve dosyalar arası örtüşme (cross-file overlap) kontrolü
    print(f"\n=== 3. TEKRAR (DUPLICATE) VE DOSYALAR ARASI ÖRTÜŞME KONTROLÜ ===")
    h_tr = pd.util.hash_pandas_object(df_tr[feat_cols], index=False).values
    h_va = pd.util.hash_pandas_object(df_va[feat_cols], index=False).values

    tr_unique = len(np.unique(h_tr))
    va_unique = len(np.unique(h_va))
    tr_dup_pct = (1.0 - tr_unique / len(h_tr)) * 100.0
    va_dup_pct = (1.0 - va_unique / len(h_va)) * 100.0

    tr_set = set(h_tr.tolist())
    va_in_tr_mask = np.isin(h_va, list(tr_set))
    cross_overlap_rows = int(va_in_tr_mask.sum())
    cross_overlap_pct = cross_overlap_rows / len(h_va) * 100.0

    # Örtüşen satırların hangi sınıflardan geldiği (ilk 8)
    print(f"Merged01 içi tekrar oranı      : %{tr_dup_pct:.2f} (Tekil: {tr_unique:,} / {len(h_tr):,})")
    print(f"Merged07 içi tekrar oranı      : %{va_dup_pct:.2f} (Tekil: {va_unique:,} / {len(h_va):,})")
    print(f"Merged07'de Merged01 ile birebir aynı özellik vektörüne sahip satır: {cross_overlap_rows:,} (%{cross_overlap_pct:.2f})")

    if cross_overlap_rows > 0:
        print("\n--- Dosyalar Arası Örtüşen Satırların En Sık 8 Sınıfı ---")
        ov_vc = df_va.loc[va_in_tr_mask, label_col].value_counts().head(8)
        for cls_name, cnt in ov_vc.items():
            cls_tot = int((df_va[label_col] == cls_name).sum())
            print(f"  {cls_name:<28}: {cnt:>7,} / {cls_tot:>7,} (%{cnt / cls_tot * 100:.1f})")

    # Etiket çelişkisi kontrolü (Merged01 içinde aynı özellik vektörü hem BENIGN hem Attack mı?)
    y_tr_bin = (df_tr[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(np.int8).values
    tmp = pd.DataFrame({"h": h_tr, "yb": y_tr_bin})
    conflict_hashes = tmp.groupby("h")["yb"].nunique()
    n_conflict_patterns = int((conflict_hashes > 1).sum())
    n_conflict_rows = int(tmp["h"].isin(conflict_hashes[conflict_hashes > 1].index).sum())
    print(f"\nMerged01 içi Binary etiket çelişkisi: {n_conflict_patterns:,} örüntü ({n_conflict_rows:,} satır, %{n_conflict_rows / len(df_tr) * 100:.3f})")

    print(f"\nKalite denetimi tamamlandı ({time.perf_counter() - t0:.2f} sn).")


if __name__ == "__main__":
    main()