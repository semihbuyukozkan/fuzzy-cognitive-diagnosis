"""Bulanık operatörler: t-norm (VE) ve t-conorm (VEYA).

Bulanık mantıkta bir önerme 0/1 yerine 0 ile 1 arasında bir doğruluk derecesine
sahiptir (ör. "öğrenci bu konuyu biliyor" = 0.8). İki dereceyi birleştirmek için:

  t-norm   T(a, b) : bulanık VE   (a ve b birlikte doğru)
  t-conorm S(a, b) : bulanık VEYA (a ya da b doğru)

Üç klasik aile (K1 katkısında karşılaştırılacak):
  "godel"       : T = min(a, b)            S = max(a, b)
  "product"     : T = a * b                S = a + b - a*b
  "lukasiewicz" : T = max(0, a + b - 1)    S = min(1, a + b)

Üçü de VE için T(a,1)=a, VEYA için S(a,0)=a özelliğini sağlar ("nötr eleman");
bu yüzden ilgisiz becerileri maskelerken VE'de 1, VEYA'da 0 ile doldururuz.

Makale karşılığı: objektif soru = bağlayıcı (VE, ∩, Eş. 3/5),
subjektif soru = telafi edici (VEYA, ∪, Eş. 6).
"""
import numpy as np

KINDS = ("godel", "product", "lukasiewicz")


def _check(kind):
    if kind not in KINDS:
        raise ValueError(f"Bilinmeyen operatör '{kind}', seçenekler: {KINDS}")


def t_norm(a, b, kind="godel"):
    """İki dereceyi bulanık VE ile birleştirir."""
    _check(kind)
    if kind == "godel":
        return np.minimum(a, b)
    if kind == "product":
        return a * b
    return np.maximum(0.0, a + b - 1.0)


def t_conorm(a, b, kind="godel"):
    """İki dereceyi bulanık VEYA ile birleştirir."""
    _check(kind)
    if kind == "godel":
        return np.maximum(a, b)
    if kind == "product":
        return a + b - a * b
    return np.minimum(1.0, a + b)


def and_reduce(x, kind="godel", axis=-1):
    """Bir eksen boyunca VE: T(x1, x2, ..., xn). Maskelenenler 1 ile doldurulmuş olmalı."""
    _check(kind)
    if kind == "godel":
        return x.min(axis=axis)
    if kind == "product":
        return x.prod(axis=axis)
    # Łukasiewicz: max(0, Σ(x_i - 1) + 1); x_i = 1 olanlar toplama 0 katar (nötr)
    return np.maximum(0.0, (x - 1.0).sum(axis=axis) + 1.0)


def or_reduce(x, kind="godel", axis=-1):
    """Bir eksen boyunca VEYA: S(x1, ..., xn). Maskelenenler 0 ile doldurulmuş olmalı."""
    _check(kind)
    if kind == "godel":
        return x.max(axis=axis)
    if kind == "product":
        return 1.0 - (1.0 - x).prod(axis=axis)
    return np.minimum(1.0, x.sum(axis=axis))
