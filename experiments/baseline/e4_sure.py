"""E4: eğitim süresi ölçümü (TEK işlem, tek iş parçacığı, paralel koşu yok).

Önceki E1-K2 koşuları 10 işlemi aynı anda çalıştırdığı için oradaki `sure_sn` sütunları birbirleriyle
kıyaslanamaz. Burada her koşu sırayla ve yalnızca bir işlemle yapılır; MH modelleri TEK zincir
(n_restart=1) ile ölçülür. (E1'de FrcSub MH modelleri 3 başlangıç kullandığı için orada süre yaklaşık 3 katıdır.)
Süre = yalnızca model eğitimi + tahmin (veri yükleme hariç), time.time() ile.

Çalıştırma:  python3 -m experiments.baseline.e4_sure
Çıktı:       results/tables/e4_sure_raw.csv, e4_sure_summary.csv
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"          # numpy/scipy import edilmeden önce
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path

import pandas as pd

from src.experiment import MfcdSpec, run_one

TABLO = Path(__file__).resolve().parents[2] / "results" / "tables"
SEEDS = [0, 1, 2]
ORAN = 0.6

MODELLER = {
    "frcsub": [("IRT-2PL", None), ("DINA", None),
               ("FuzzyCDF-orig", MfcdSpec(learn_mu=False)), ("MFCD-core", MfcdSpec(learn_mu=True))],
    "synthetic": [("IRT-2PL", None), ("DINA", None),
                  ("FuzzyCDF-orig", MfcdSpec(learn_mu=False)), ("MFCD-core", MfcdSpec(learn_mu=True)),
                  ("MFCD hepsi (süre+zaman+μ)", MfcdSpec(learn_mu=True, use_time=True, use_duration=True))],
}


def main():
    satirlar = []
    for veri, liste in MODELLER.items():
        for ad, spec in liste:
            for s in SEEDS:
                r = run_one(veri, "MFCD-core" if ad.startswith("MFCD hepsi") else ad, ORAN, s, spec)
                satirlar.append({"veri": veri, "model": ad, "seed": s, "sure_sn": r["sure_sn"], "F1": r["F1"]})
                print(f"{veri:10s} {ad:28s} seed={s} {r['sure_sn']:8.2f} sn", flush=True)
    df = pd.DataFrame(satirlar)
    TABLO.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLO / "e4_sure_raw.csv", index=False)
    ozet = df.groupby(["veri", "model"], sort=False).sure_sn.agg(["mean", "std", "min", "max"]).round(2)
    ozet.to_csv(TABLO / "e4_sure_summary.csv")
    print(ozet.to_string())


if __name__ == "__main__":
    main()
