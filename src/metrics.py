"""Değerlendirme metrikleri.

Doğru cevap (r = 1) "pozitif" sınıf kabul edilir; tahmin = (P >= 0.5).
  PRE    (precision): "doğru" dediklerimizin kaçı gerçekten doğru   = TP / (TP + FP)
  Recall            : gerçek doğruların kaçını yakaladık             = TP / (TP + FN)
  F1                : PRE ve Recall'ın harmonik ortalaması
  MAE, RMSE         : olasılık tahmini ile gerçek 0/1 cevap arasındaki hata
  AUC               : EŞİKTEN BAĞIMSIZ sıralama başarısı: rastgele seçilen bir doğru cevabın olasılığı,
                      rastgele seçilen bir yanlış cevabın olasılığından yüksek mi (0.5 = şans, 1 = mükemmel)
  ort_P, ort_R      : test hücrelerinde ortalama tahmin olasılığı ve gerçek doğru oranı. Aralarındaki fark
                      kalibrasyon kaymasıdır (model sistematik olarak "doğru" demeye mi kayıyor?)
Payı 0 olan oranlar 0 sayılır.
"""
import numpy as np
from scipy.stats import rankdata


def _bol(a, b):
    return float(a / b) if b > 0 else 0.0


def auc(y, p):
    """Rank tabanlı AUC (Mann-Whitney U); eşit skorlar ortalama sıra alır. Tek sınıf varsa 0.5."""
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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
        "AUC": auc(y, p),
        "ort_P": float(p.mean()),
        "ort_R": float(y.mean()),
    }
