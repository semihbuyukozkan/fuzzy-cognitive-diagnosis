"""MCMC yakınsama tanıları.

Split R-hat (Gelman-Rubin; BDA3 §11.4): aynı posterior'dan, farklı başlangıçlı C zincir çalıştırılır.
Her zincir ikiye bölünür (zincir içi sürüklenmeyi de yakalamak için). Sonra
    W = zincir içi varyansların ortalaması,  B = zincir ortalamalarının varyansı (x n)
    R-hat = sqrt( ((n-1)/n · W + B/n) / W )
Zincirler aynı yere yakınsadıysa B küçüktür ve R-hat ≈ 1. Pratik eşik: R-hat < 1.1 (daha sıkı: 1.05).
"""
import numpy as np


def split_rhat(draws):
    """draws: (zincir, örnek, parametre) -> R-hat (parametre,). Örnek sayısı en az 4 olmalı."""
    d = np.asarray(draws, dtype=float)
    if d.ndim == 2:
        d = d[:, :, None]
    c, n, _ = d.shape
    h = n // 2
    z = np.concatenate([d[:, :h], d[:, h:2 * h]], axis=0)            # (2C, h, P)
    W = z.var(axis=1, ddof=1).mean(axis=0)
    B = h * z.mean(axis=1).var(axis=0, ddof=1)
    var_plus = (h - 1) / h * W + B / h
    return np.sqrt(var_plus / np.maximum(W, 1e-300))


def ozet(rhat):
    """R-hat vektörü için özet: ortanca, maksimum, 1.1 ve 1.05 üstü sayısı."""
    r = np.asarray(rhat)
    return {"n": int(r.size), "medyan": float(np.median(r)), "maks": float(r.max()),
            "n_ust_1.1": int((r > 1.1).sum()), "n_ust_1.05": int((r > 1.05).sum())}
