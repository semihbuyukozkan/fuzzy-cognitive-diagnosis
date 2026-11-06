"""K2: faktör ablation'ı (YALNIZCA SENTETİK veri). Her ayar sıfırdan eğitilir; aynı seed ve bölme.

Veri varyantları: "synthetic" (zaman/süre etkili) ve "synthetic_null" (NEGATİF KONTROL: zaman/süre etkisiz).
Eğitim oranı 0.6, 10 seed. Operatörler Gödel (min/max). Referans: "hiçbiri" (FuzzyCDF-orig mantığı).

Ayarlar:
  hiçbiri                       : μ = Q, ağırlıksız
  +bilgi açığı                  : dereceli μ öğrenilir
  +zaman (olabilirlik)          : Eş. 2 ağırlığı gözlemin olabilirlik ağırlığı (bizim varsayımımız)
  +zaman (Eş.3 çarpımı)         : yeterlilik, VE/VEYA'dan önce (T-Δt)/T ile çarpılır (makalenin yazımı)
  +bilgi açığı +zaman (olab.)
  +süre (bizim yorum)           : Eş. 7, [0,1]'e ölçekli ve w = ω(1+λ·ωC) (makale normalize etmiyor)
  hepsi                         : bilgi açığı + zaman (olab.) + süre

Çalıştırma:  python3 -m experiments.proposed.k2_ablation
"""
from pathlib import Path

import pandas as pd

from src.analysis import eslestirilmis_fark, ozet_tablo
from src.experiment import SEEDS, MfcdSpec, run_grid

TABLO = Path(__file__).resolve().parents[2] / "results" / "tables"
AYARLAR = {
    "hiçbiri": MfcdSpec(),
    "+bilgi açığı": MfcdSpec(learn_mu=True),
    "+zaman (olabilirlik)": MfcdSpec(use_time=True),
    "+zaman (Eş.3 çarpımı)": MfcdSpec(use_time=True, time_mode="eta"),
    "+bilgi açığı +zaman": MfcdSpec(learn_mu=True, use_time=True),
    "+süre (bizim yorum)": MfcdSpec(use_duration=True),
    "hepsi": MfcdSpec(learn_mu=True, use_time=True, use_duration=True),
}
VERILER = ["synthetic", "synthetic_null"]


def main():
    jobs = [(v, ad, 0.6, s, sp) for v in VERILER for ad, sp in AYARLAR.items() for s in SEEDS]
    jobs += [(v, m, 0.6, s) for v in VERILER for m in ("IRT-2PL", "DINA") for s in SEEDS]
    print(f"{len(jobs)} koşu...")
    df = pd.DataFrame(run_grid(jobs)).sort_values(["veri", "model", "seed"])
    TABLO.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLO / "k2_raw.csv", index=False)
    ozet = ozet_tablo(df, ["veri", "model", "oran"])
    ozet.to_csv(TABLO / "k2_summary.csv", index=False)
    fark = pd.concat([eslestirilmis_fark(df, "hiçbiri", m, grup=("veri", "oran"), yuksek_iyi=y)
                      for m, y in (("F1", True), ("RMSE", False))])
    fark.to_csv(TABLO / "k2_paired.csv", index=False)
    pd.set_option("display.width", 220)
    print(ozet[["veri", "model", "F1_ort", "F1_std", "RMSE_ort", "RMSE_std", "sure_sn_ort"]].round(4).to_string(index=False))
    print(fark.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
