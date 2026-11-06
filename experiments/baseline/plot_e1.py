"""E1 grafikleri: eğitim oranına göre F1 ve RMSE (ortalama ± std, 10 seed).

Çalıştırma:  python3 -m experiments.baseline.plot_e1
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.plotting import INK2, cizgi_hata, kaydet

KOK = Path(__file__).resolve().parents[2]
MODELLER = ["IRT-2PL", "DINA", "FuzzyCDF-orig", "MFCD-core"]


def main():
    s = pd.read_csv(KOK / "results" / "tables" / "e1_frcsub_summary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    cizgi_hata(axs[0], s, "oran", "F1", "model", MODELLER, "F1 (yüksek iyi)", "F1")
    cizgi_hata(axs[1], s, "oran", "RMSE", "model", MODELLER, "RMSE (düşük iyi)", "RMSE")
    axs[0].legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.suptitle("FrcSub: eğitim oranına göre performans (ortalama ± std, 10 seed)",
                 x=0.01, ha="left", fontsize=12)
    (KOK / "results" / "figures").mkdir(parents=True, exist_ok=True)
    kaydet(fig, KOK / "results" / "figures" / "e1_frcsub_f1_rmse.png")
    print("kaydedildi: results/figures/e1_frcsub_f1_rmse.png")


if __name__ == "__main__":
    main()
