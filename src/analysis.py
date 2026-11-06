"""Sonuç tablolarını özetleyen yardımcılar."""
import numpy as np
import pandas as pd

METRIKLER = ["PRE", "Recall", "F1", "MAE", "RMSE", "AUC", "ort_P", "ort_R", "sure_sn"]


def ozet_tablo(df, grup):
    """grup sütunlarına göre metriklerin ortalama ve std değerleri."""
    g = df.groupby(grup)[[m for m in METRIKLER if m in df.columns]]
    out = g.mean().add_suffix("_ort").join(g.std().add_suffix("_std"))
    out["n_seed"] = g.size()
    return out.reset_index()


def eslestirilmis_fark(df, referans, metrik="F1", grup=("veri", "oran"), yuksek_iyi=True):
    """Her model için, AYNI seed'deki referans modelle fark (model − referans).

    Aynı seed = aynı veri ve aynı eğitim/test bölmesi, bu yüzden eşleştirilmiş fark adil karşılaştırmadır.
    Dönüş: ortalama fark, std, modelin referanstan iyi olduğu seed sayısı ve toplam seed.
    """
    satirlar = []
    for anahtar, d in df.groupby(list(grup)):
        p = d.pivot(index="seed", columns="model", values=metrik)
        if referans not in p:
            continue
        for m in p.columns:
            if m == referans:
                continue
            fark = (p[m] - p[referans]).dropna()
            iyi = (fark > 0) if yuksek_iyi else (fark < 0)
            satirlar.append({**dict(zip(grup, anahtar if isinstance(anahtar, tuple) else (anahtar,))),
                             "model": m, "metrik": metrik, "fark_ort": fark.mean(),
                             "fark_std": fark.std(), "daha_iyi_seed": int(iyi.sum()), "n_seed": len(fark)})
    return pd.DataFrame(satirlar)
