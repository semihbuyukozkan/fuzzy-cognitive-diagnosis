"""Değerlendirme metrikleri.

Doğru cevap (r = 1) "pozitif" sınıf kabul edilir; tahmin = (P >= 0.5).
  PRE    (precision): "doğru" dediklerimizin kaçı gerçekten doğru   = TP / (TP + FP)
  Recall            : gerçek doğruların kaçını yakaladık             = TP / (TP + FN)
  F1                : PRE ve Recall'ın harmonik ortalaması
  MAE, RMSE         : olasılık tahmini ile gerçek 0/1 cevap arasındaki hata
Payı 0 olan oranlar 0 sayılır.
"""
import numpy as np


def _bol(a, b):
    return float(a / b) if b > 0 else 0.0


def evaluate(R, P, mask, threshold=0.5):
    """mask=True olan (test) hücreler üzerinde metrikleri hesaplar."""
    y, p = R[mask].astype(float), P[mask]
    pred = (p >= threshold).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    pre, rec = _bol(tp, tp + fp), _bol(tp, tp + fn)
    return {
        "PRE": pre,
        "Recall": rec,
        "F1": _bol(2 * pre * rec, pre + rec),
        "MAE": float(np.abs(p - y).mean()),
        "RMSE": float(np.sqrt(((p - y) ** 2).mean())),
    }
