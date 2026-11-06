"""K1 döngüsellik kontrolü: sentetik veriyi ÇARPIM operatörüyle üretip K1'i tekrarlar.

Soru: Gödel verisinde Gödel kazanmıştı (üreteç = model). Çarpımla üretilmiş veride hangi operatör kazanıyor?
Kazanan çarpım olursa "operatör sıralaması üreteci izliyor" (döngüsellik) kanıtlanır.
Ayarlar ve protokol K1 sentetik kısmıyla aynı (5 ayar, %60, 10 seed, zaman/süre kapalı, μ=Q).
Fark referansı: Gödel/Gödel koşusu (K1 ile aynı); çarpım ayarları da tabloda görülür.

Çalıştırma:  python3 -m experiments.proposed.k1_product_veri
Çıktı:       results/tables/k1_product_{raw,summary,paired}.csv
"""
from pathlib import Path

import pandas as pd

from src.analysis import eslestirilmis_fark, ozet_tablo
from src.experiment import SEEDS, MfcdSpec, run_grid

TABLO = Path(__file__).resolve().parents[2] / "results" / "tables"
AILELER = ["godel", "product", "lukasiewicz"]


def main():
    ayarlar = [(a, "godel") for a in AILELER] + [("godel", o) for o in AILELER[1:]]
    jobs = [("synthetic_product", f"VE={a} / VEYA={o}", 0.6, s, MfcdSpec(and_kind=a, or_kind=o))
            for a, o in ayarlar for s in SEEDS]
    print(f"{len(jobs)} koşu...")
    df = pd.DataFrame(run_grid(jobs)).sort_values(["model", "seed"])
    df.to_csv(TABLO / "k1_product_raw.csv", index=False)
    ozet_tablo(df, ["veri", "model", "oran"]).to_csv(TABLO / "k1_product_summary.csv", index=False)
    g = pd.concat([eslestirilmis_fark(df, "VE=godel / VEYA=godel", m, yuksek_iyi=y)
                   for m, y in (("F1", True), ("RMSE", False), ("AUC", True))])
    g.to_csv(TABLO / "k1_product_paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(ozet_tablo(df, ["veri", "model", "oran"])[["model", "F1_ort", "F1_std", "RMSE_ort", "AUC_ort"]].round(4).to_string(index=False))
    print(g.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
