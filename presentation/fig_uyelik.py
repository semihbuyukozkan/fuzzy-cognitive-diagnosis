"""Sunum görseli: sigmoid üyelik fonksiyonu (Eş. 1) ve α, β'nın etkisi. Kendi kodumuzdan (src/memberships.py) çizilir.
Çalıştırma:  python3 -m presentation.fig_uyelik   ->  presentation/fig_uyelik.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.memberships import membership
from src.plotting import GRID, INK, INK2, RENKLER, SURFACE, kaydet

KOK = Path(__file__).resolve().parents[1]


def main():
    theta = np.linspace(-3, 3, 200)
    ayar = [("kolay beceri (β = −1, α = 1)", 1.0, -1.0), ("orta beceri (β = 0, α = 1)", 1.0, 0.0),
            ("zor ve keskin beceri (β = 1, α = 3)", 3.0, 1.0)]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.set_facecolor(SURFACE)
    for i, (ad, a, b) in enumerate(ayar):
        m = membership(theta, np.array([a]), np.array([b]))[:, 0]
        ax.plot(theta, m, color=RENKLER[i], linewidth=2.6, label=ad)
    ax.axhline(0.5, color=GRID, linewidth=1)
    ax.set_xlabel("öğrencinin yeteneği θ", color=INK2, fontsize=12)
    ax.set_ylabel("üyelik derecesi a  (becerinin bilinme derecesi)", color=INK2, fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    kaydet(fig, KOK / "presentation" / "fig_uyelik.png")
    print("kaydedildi")


if __name__ == "__main__":
    main()
