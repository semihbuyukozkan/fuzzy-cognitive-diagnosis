"""K1 grafiği: Gödel'e (min/max) göre eşleştirilmiş fark (aynı seed ve bölme), ortalama ± std (10 seed).

Sıfır çizgisi = Gödel. F1 için yukarı iyi, RMSE için aşağı iyi.
Çalıştırma:  python3 -m experiments.proposed.plot_k1
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.plotting import GRID, INK, INK2, RENKLER, SURFACE, kaydet

KOK = Path(__file__).resolve().parents[2]
RENK = {"product": RENKLER[1], "lukasiewicz": RENKLER[2]}       # sabit sıra: aile -> renk
ISARET = {"VE": "o", "VEYA": "s"}


def _stil(ax, baslik, ylabel):
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.set_title(baslik, color=INK, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main():
    p = pd.read_csv(KOK / "results" / "tables" / "k1_paired.csv")
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.5))
    for r, (met, yon) in enumerate((("F1", "F1 farkı (yukarı iyi)"), ("RMSE", "RMSE farkı (aşağı iyi)"))):
        # sol: FrcSub, yalnızca VE; x = eğitim oranı
        ax = axs[r, 0]
        d = p[(p.veri == "frcsub") & (p.metrik == met)]
        for i, fam in enumerate(["product", "lukasiewicz"]):
            x = d[d.model == f"VE={fam}"].sort_values("oran")
            ax.errorbar(x.oran + (i - 0.5) * 0.012, x.fark_ort, yerr=x.fark_std, color=RENK[fam],
                        marker="o", markersize=6, linewidth=1.6, capsize=3, label=f"VE={fam}",
                        markeredgecolor=SURFACE)
        _stil(ax, f"FrcSub (yalnızca VE), {met}: Gödel'e göre fark", yon)
        ax.set_xticks([0.5, 0.6, 0.7]); ax.set_xlabel("Eğitim oranı", color=INK2)
        if r == 0:
            ax.legend(frameon=False, labelcolor=INK2, fontsize=8)
        # sağ: sentetik; x = değiştirilen operatör
        ax = axs[r, 1]
        d = p[(p.veri == "synthetic") & (p.metrik == met)].copy()
        etiketler = []
        for k, (_, row) in enumerate(d.iterrows()):
            kat = "VE" if row.model.startswith("VE=") and "VEYA=godel" in row.model else "VEYA"
            fam = row.model.split("VE=")[1].split(" ")[0] if kat == "VE" else row.model.split("VEYA=")[1]
            ax.errorbar([k], [row.fark_ort], yerr=[row.fark_std], color=RENK[fam], marker=ISARET[kat],
                        markersize=7, capsize=3, linewidth=1.6, markeredgecolor=SURFACE)
            etiketler.append(f"{kat}={fam}")
        _stil(ax, f"Sentetik (oran 0.6), {met}: Gödel'e göre fark", yon)
        ax.set_xticks(range(len(etiketler))); ax.set_xticklabels(etiketler, rotation=15, fontsize=8)
    fig.suptitle("K1: operatör karşılaştırması (eşleştirilmiş fark, 10 seed)", x=0.01, ha="left", fontsize=12)
    (KOK / "results" / "figures").mkdir(parents=True, exist_ok=True)
    kaydet(fig, KOK / "results" / "figures" / "k1_operators.png")
    print("kaydedildi: results/figures/k1_operators.png")


if __name__ == "__main__":
    main()
