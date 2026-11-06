"""E1: FrcSub üzerinde baseline'lar + MFCD-core, 6 eğitim oranı x 10 seed.

Çalıştırma:  python3 -m experiments.baseline.e1_frcsub
Çıktılar:    results/tables/e1_frcsub_raw.csv      (her koşu)
             results/tables/e1_frcsub_summary.csv  (oran/model bazında ortalama ± std)
"""
from pathlib import Path

import pandas as pd

from src.experiment import RATIOS, SEEDS, run_grid

MODELLER = ["IRT-2PL", "DINA", "FuzzyCDF-orig", "MFCD-core"]
TABLO = Path(__file__).resolve().parents[2] / "results" / "tables"


def ozet(df):
    metrikler = ["PRE", "Recall", "F1", "MAE", "RMSE", "sure_sn"]
    g = df.groupby(["model", "oran"])[metrikler]
    out = g.mean().add_suffix("_ort").join(g.std().add_suffix("_std")).reset_index()
    out["n_seed"] = g.size().values
    return out


def main():
    jobs = [("frcsub", m, r, s) for m in MODELLER for r in RATIOS for s in SEEDS]
    print(f"{len(jobs)} koşu başlıyor...")
    df = pd.DataFrame(run_grid(jobs)).sort_values(["model", "oran", "seed"])
    TABLO.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLO / "e1_frcsub_raw.csv", index=False)
    s = ozet(df)
    s.to_csv(TABLO / "e1_frcsub_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s[s.oran.isin([0.5, 0.6, 0.7])][["model", "oran", "PRE_ort", "Recall_ort", "F1_ort",
                                           "MAE_ort", "RMSE_ort", "sure_sn_ort"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
