"""K1 (ANA KATKI): bulanık operatör karşılaştırması. Model sıfırdan eğitilir; aynı seed, aynı bölme,
aynı hiperparametre. Yalnızca operatör değişir. μ = Q (açık faktörü kapalı) ki fark sadece operatörden gelsin.

- FrcSub (yalnızca objektif soru): yalnızca VE (t-norm) ailesi; 3 ayar x oran {0.5,0.6,0.7} x 10 seed.
  FrcSub'da her koşu 3 başlangıçla çalışır, eğitim olabilirliği en yüksek zincir seçilir (yerel çukur önlemi).
- Sentetik: VE ailesi değişirken VEYA=Gödel; VEYA ailesi değişirken VE=Gödel (5 ayar), oran 0.6, 10 seed.
  (Zaman ve süre faktörleri kapalı; operatör etkisini izole etmek için.)

Çalıştırma:  python3 -m experiments.proposed.k1_operators
Çıktı:       results/tables/k1_*.csv
"""
from pathlib import Path

import pandas as pd

from src.analysis import eslestirilmis_fark, ozet_tablo
from src.experiment import SEEDS, MfcdSpec, run_grid

TABLO = Path(__file__).resolve().parents[2] / "results" / "tables"
AILELER = ["godel", "product", "lukasiewicz"]


def isler():
    jobs = []
    for a in AILELER:                                              # FrcSub: yalnızca VE
        for r in (0.5, 0.6, 0.7):
            for s in SEEDS:
                jobs.append(("frcsub", f"VE={a}", r, s, MfcdSpec(and_kind=a, n_restart=3)))
    ayarlar = [(a, "godel") for a in AILELER] + [("godel", o) for o in AILELER[1:]]
    for a, o in ayarlar:                                           # sentetik: VE ve VEYA
        for s in SEEDS:
            jobs.append(("synthetic", f"VE={a} / VEYA={o}", 0.6, s, MfcdSpec(and_kind=a, or_kind=o)))
    return jobs


def main():
    jobs = isler()
    print(f"{len(jobs)} koşu...")
    df = pd.DataFrame(run_grid(jobs)).sort_values(["veri", "model", "oran", "seed"])
    TABLO.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLO / "k1_raw.csv", index=False)
    ozet_tablo(df, ["veri", "model", "oran"]).to_csv(TABLO / "k1_summary.csv", index=False)
    f = pd.concat([eslestirilmis_fark(df[df.veri == "frcsub"], "VE=godel", m, yuksek_iyi=y)
                   for m, y in (("F1", True), ("RMSE", False))])
    g = pd.concat([eslestirilmis_fark(df[df.veri == "synthetic"], "VE=godel / VEYA=godel", m, yuksek_iyi=y)
                   for m, y in (("F1", True), ("RMSE", False))])
    pd.concat([f, g]).to_csv(TABLO / "k1_paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(ozet_tablo(df, ["veri", "model", "oran"])[["veri", "model", "oran", "F1_ort", "F1_std", "RMSE_ort", "RMSE_std", "sure_sn_ort"]].round(3).to_string(index=False))
    print(pd.concat([f, g]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
