"""K2 grafiği: "hiçbiri"ye göre eşleştirilmiş fark (aynı seed ve bölme), ortalama ± std (10 seed).

Sütunlar: etkili sentetik veri | negatif kontrol (zaman/süre etkisiz). Satırlar: F1, RMSE, AUC.
Sıfır çizgisi = "hiçbiri" (FuzzyCDF-orig mantığı). F1 ve AUC için sağa iyi, RMSE için sola iyi.
Çalıştırma:  python3 -m experiments.proposed.plot_k2
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.plotting import GRID, INK, INK2, RENKLER, SURFACE, kaydet

KOK = Path(__file__).resolve().parents[2]
SIRA = ["+bilgi açığı", "+zaman (olabilirlik)", "+zaman (Eş.3 çarpımı)", "+bilgi açığı +zaman",
        "+süre (bizim yorum)", "hepsi", "IRT-2PL", "DINA"]
BASLIK = {"synthetic": "Sentetik: zaman/süre ETKİLİ", "synthetic_null": "Negatif kontrol: zaman/süre ETKİSİZ"}
METRIK = [("F1", "F1 farkı (sağa iyi)"), ("RMSE", "RMSE farkı (sola iyi)"), ("AUC", "AUC farkı (sağa iyi)")]


def main():
    p = pd.read_csv(KOK / "results" / "tables" / "k2_paired.csv")
    fig, axs = plt.subplots(3, 2, figsize=(11, 10), sharey=True)
    for r, (met, xl) in enumerate(METRIK):
        for c, veri in enumerate(["synthetic", "synthetic_null"]):
            ax = axs[r, c]
            ax.set_facecolor(SURFACE)
            d = p[(p.veri == veri) & (p.metrik == met)].set_index("model").reindex(SIRA)
            y = range(len(SIRA))
            ax.errorbar(d.fark_ort, y, xerr=d.fark_std, fmt="o", color=RENKLER[0], markersize=6,
                        capsize=3, linewidth=1.4, markeredgecolor=SURFACE)
            ax.axvline(0, color=INK2, linewidth=0.9)
            ax.set_yticks(list(y)); ax.set_yticklabels(SIRA, fontsize=8.5, color=INK2)
            ax.invert_yaxis()
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.set_xlabel(xl, color=INK2, fontsize=9)
            if r == 0:
                ax.set_title(BASLIK[veri], color=INK, fontsize=10, loc="left")
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.tick_params(colors=INK2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
    fig.suptitle('K2: "hiçbiri"ye göre eşleştirilmiş fark (ortalama ± std, 10 seed, eğitim oranı 0.6)',
                 x=0.01, ha="left", fontsize=12)
    (KOK / "results" / "figures").mkdir(parents=True, exist_ok=True)
    kaydet(fig, KOK / "results" / "figures" / "k2_ablation.png")
    print("kaydedildi: results/figures/k2_ablation.png")


if __name__ == "__main__":
    main()
