from pathlib import Path
from collections import Counter
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "raw_ciciot2023"

def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"HATA: {DATA_DIR} içinde CSV dosyası bulunamadı.")
        return

    print(f"Bulunan dosya sayısı: {len(csv_files)} ({DATA_DIR})\n")

    total_counts = Counter()
    total_rows = 0
    per_file_rows = []
    cols_ref = None
    label_col = None

    for fp in csv_files:
        # Önce sadece başlık satırını oku
        header = pd.read_csv(fp, nrows=0)
        cols = list(header.columns)
        if cols_ref is None:
            cols_ref = cols
            label_col = "Label" if "Label" in cols else cols[-1]
            print(f"Sütun sayısı: {len(cols)} | Etiket sütunu: '{label_col}'")
            print(f"İlk 8 sütun : {cols[:8]}\n")

        # RAM tasarrufu için yalnızca etiket sütununu oku
        s = pd.read_csv(fp, usecols=[label_col])[label_col].astype(str).str.strip()
        vc = s.value_counts()
        n = len(s)
        total_rows += n
        total_counts.update(vc.to_dict())

        benign_n = int(vc.get("BenignTraffic", 0) + vc.get("Benign", 0) + vc.get("BENIGN", 0))
        top3 = ", ".join(f"{k}:{v:,}" for k, v in vc.head(3).items())

        per_file_rows.append({
            "dosya": fp.name,
            "satir": n,
            "sinif_sayisi": len(vc),
            "benign_satir": benign_n,
            "benign_pct": round(benign_n / n * 100, 2) if n else 0.0,
            "en_sik_3_sinif": top3,
        })

    print("=== DOSYA BAZLI ÖZET ===")
    print(pd.DataFrame(per_file_rows).to_string(index=False))

    print(f"\n=== TOPLAM SINIF DAĞILIMI ({total_rows:,} SATIR, {len(total_counts)} SINIF) ===")
    dist_df = pd.DataFrame(
        [(k, v, round(v / total_rows * 100, 3)) for k, v in total_counts.most_common()],
        columns=["Sinif", "Satir", "Oran_%"],
    )
    print(dist_df.to_string(index=False))

if __name__ == "__main__":
    main()