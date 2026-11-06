"""MCMC yakınsama kanıtı: 4 zincir (dağınık başlangıç), split R-hat, log-olabilirlik izi.

Her (veri, model) için iki başlangıç türüyle 4'er zincir, 3000 adım (ilk 1500 burn-in), her 5. örnek:
  - "dagınık": birbirinden çok farklı başlangıçlar (R-hat için gerçek sınav)
  - "uretim" : deney koşularında kullandığımız mantıklı başlangıç (üretim koşuları böyle başlıyor)
Aynı veri ve aynı eğitim/test bölmesi; zincirlerin seed'leri farklı. Her zincir için test F1/RMSE de
kaydedilir: parametreler yakınsamasa bile tahminler zincirler arasında tutarlı mı?

Çalıştırma:  python3 -m experiments.proposed.mcmc_convergence
Çıktı:       results/tables/mcmc_convergence.csv, mcmc_chain_metrics.csv, results/figures/mcmc_trace.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diagnostics import ozet, split_rhat
from src.experiment import run_chains
from src.plotting import GRID, INK, INK2, RENKLER, kaydet

KOK = Path(__file__).resolve().parents[2]
VERI_MODEL = [("frcsub", "FuzzyCDF-orig"), ("frcsub", "MFCD-core"),
              ("synthetic", "FuzzyCDF-orig"), ("synthetic", "MFCD-core")]
CHAIN_SEEDS = [100, 101, 102, 103]
GRUPLAR = ["ll", "theta", "alpha", "beta", "slip", "guess", "mu"]


def main():
    jobs = [(v, m, 0.6, 0, c, 3000, 1500, None, disp)
            for disp in (True, False) for v, m in VERI_MODEL for c in CHAIN_SEEDS]
    print(f"{len(jobs)} zincir koşuyor...")
    sonuc = run_chains(jobs)
    satirlar, izler, zincir_met = [], {}, []
    for disp in (True, False):
        etiket = "dagınık" if disp else "uretim"
        for v, m in VERI_MODEL:
            zs = [r for r in sonuc if r["veri"] == v and r["model"] == m and r["dispersed"] == disp]
            if disp:
                izler[(v, m)] = (np.array([z["ll_trace"] for z in zs]), zs[0]["burn"])
            for g in GRUPLAR:
                if g == "mu" and m == "FuzzyCDF-orig":
                    continue                      # bu modelde μ = Q sabit
                d = np.stack([z["samples"][g].reshape(len(z["samples"][g]), -1) for z in zs])
                satirlar.append({"init": etiket, "veri": v, "model": m, "grup": g, **ozet(split_rhat(d))})
            for z in zs:
                zincir_met.append({"init": etiket, "veri": v, "model": m, "zincir": z["zincir"],
                                   "F1": z["F1"], "RMSE": z["RMSE"], "ll_ort": z["ll_ort"],
                                   "ll_surukleme": z["ll_drift"]})
    df = pd.DataFrame(satirlar)
    zm = pd.DataFrame(zincir_met)
    (KOK / "results" / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(KOK / "results" / "tables" / "mcmc_convergence.csv", index=False)
    zm.to_csv(KOK / "results" / "tables" / "mcmc_chain_metrics.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    print(zm.round(4).to_string(index=False))
    izle_ciz(izler)


def izle_ciz(izler):
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, ((v, m), (tr, burn)) in zip(axs.ravel(), izler.items()):
        ax.set_facecolor("#fcfcfb")
        for i, t in enumerate(tr):
            ax.plot(t, color=RENKLER[i], linewidth=0.9, label=f"Zincir {i + 1}")
        son = tr[:, burn:]
        pay = (son.max() - son.min()) * 1.5
        ax.set_ylim(son.min() - pay, son.max() + pay)      # burn-in başı eksen dışında kalabilir
        ax.axvline(burn, color=INK2, linewidth=0.8, linestyle="--")
        ax.set_title(f"{v} / {m}", color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    for ax in axs[1]:
        ax.set_xlabel("MH adımı (kesikli çizgi: burn-in sonu)", color=INK2)
    for ax in axs[:, 0]:
        ax.set_ylabel("Log-olabilirlik (eğitim)", color=INK2)
    axs[0, 0].legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower right")
    fig.suptitle("MH zincirleri: log-olabilirlik izi (4 dağınık başlangıç)", x=0.01, ha="left", fontsize=12)
    (KOK / "results" / "figures").mkdir(parents=True, exist_ok=True)
    kaydet(fig, KOK / "results" / "figures" / "mcmc_trace.png")


if __name__ == "__main__":
    main()
