"""Ortak grafik yardımcıları (kendi çizimlerimiz; makale şekilleri kullanılmaz).

Renkler: doğrulanmış kategorik palet, sabit sırayla (mavi, turuncu, aqua, sarı, ...).
Kontrastı düşük renkler (aqua, sarı) için ikincil kodlama: farklı işaret şekilleri + legend;
tablo görünümü results/tables/ altındaki CSV dosyalarıdır.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RENKLER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ISARETLER = ["o", "s", "^", "D", "v", "P", "X", "*"]


def cizgi_hata(ax, ozet, x, y, grup, sirala, baslik, ylabel, xlabel="Eğitim oranı"):
    """Her grup için ortalama çizgi + ±1 std hata çubuğu. Renk gruba (sabit sıra) bağlıdır."""
    ax.set_facecolor(SURFACE)
    for i, g in enumerate(sirala):
        d = ozet[ozet[grup] == g].sort_values(x)
        ax.errorbar(d[x], d[f"{y}_ort"], yerr=d[f"{y}_std"], color=RENKLER[i], marker=ISARETLER[i],
                    markersize=6, linewidth=1.8, capsize=3, elinewidth=1, label=g,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_title(baslik, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def kaydet(fig, yol):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(yol, dpi=200, facecolor=SURFACE)
    plt.close(fig)
