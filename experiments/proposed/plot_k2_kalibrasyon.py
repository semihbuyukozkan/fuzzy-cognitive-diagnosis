"""K2 kalibrasyon grafiği: ortalama tahmin olasılığı − gerçek doğru oranı (10 seed ortalaması).

Sıfır = iyi kalibre. Pozitif = model gereğinden fazla "doğru" diyor.
F1 yalnızca doğru sınıfı ödüllendirdiği için bu kayma F1'i şişirebilir (bkz. docs/report_notes.md).
Çalıştırma:  python3 -m experiments.proposed.plot_k2_kalibrasyon
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.plotting import GRID, INK, INK2, RENKLER, SURFACE, kaydet

KOK = Path(__file__).resolve().parents[2]
SIRA = ["hiçbiri", "+bilgi açığı", "+zaman (olabilirlik)", "+zaman (Eş.3 çarpımı)",
        "+bilgi açığı +zaman", "+süre (bizim yorum)", "hepsi", "IRT-2PL", "DINA"]
BASLIK = {"synthetic": "Zaman/süre ETKİLİ veri", "synthetic_null": "Negatif kontrol (etkisiz)"}


def main():
    s = pd.read_csv(KOK / "results" / "tables" / "k2_summary.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, veri in zip(axs, ["synthetic", "synthetic_null"]):
        ax.set_facecolor(SURFACE)
        d = s[s.veri == veri].set_index("model").reindex(SIRA)
        fark = d.ort_P_ort - d.ort_R_ort
        y = range(len(SIRA))
        ax.barh(list(y), fark, color=RENKLER[0], height=0.55)
        for yi, f in zip(y, fark):                       # değer etiketi (erişilebilirlik)
            ax.text(f + (0.004 if f >= 0 else -0.004), yi, f"{f:+.3f}", va="center",
                    ha="left" if f >= 0 else "right", fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, linewidth=0.9)
        ax.set_yticks(list(y)); ax.set_yticklabels(SIRA, fontsize=9, color=INK2)
        ax.set_xlim(-0.08, 0.14)
        ax.set_xlabel("ort. tahmin olasılığı − gerçek doğru oranı", color=INK2, fontsize=9)
        ax.set_title(BASLIK[veri], color=INK, fontsize=10, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axs[0].invert_yaxis()                      # sharey: yalnızca bir kez çevrilir
    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle("K2 kalibrasyon: sıfırdan sapma = sistematik kayma (eğitim oranı 0.6, 10 seed)",
                 x=0.01, ha="left", fontsize=12)
    kaydet(fig, KOK / "results" / "figures" / "k2_kalibrasyon.png")
    print("kaydedildi: results/figures/k2_kalibrasyon.png")


if __name__ == "__main__":
    main()
