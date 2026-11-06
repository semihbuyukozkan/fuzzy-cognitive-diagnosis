"""Operatörler için elle hesaplanmış küçük örnekler."""
import numpy as np
import pytest

from src.fuzzy_ops import and_reduce, or_reduce, t_conorm, t_norm

A, B = 0.8, 0.6


@pytest.mark.parametrize("kind,vee,veya", [
    ("godel", 0.6, 0.8),
    ("product", 0.48, 0.92),            # 0.8*0.6 ; 0.8+0.6-0.48
    ("lukasiewicz", 0.4, 1.0),          # max(0,1.4-1) ; min(1,1.4)
])
def test_iki_degerli_ornek(kind, vee, veya):
    assert t_norm(A, B, kind) == pytest.approx(vee)
    assert t_conorm(A, B, kind) == pytest.approx(veya)


@pytest.mark.parametrize("kind", ["godel", "product", "lukasiewicz"])
def test_notr_elemanlar(kind):
    assert t_norm(0.37, 1.0, kind) == pytest.approx(0.37)
    assert t_conorm(0.37, 0.0, kind) == pytest.approx(0.37)


@pytest.mark.parametrize("kind", ["godel", "product", "lukasiewicz"])
def test_reduce_ikili_fonksiyonla_tutarli(kind):
    x = np.array([0.9, 0.7, 0.5])
    ve = t_norm(t_norm(x[0], x[1], kind), x[2], kind)
    veya = t_conorm(t_conorm(x[0], x[1], kind), x[2], kind)
    assert and_reduce(x, kind) == pytest.approx(ve)
    assert or_reduce(x, kind) == pytest.approx(veya)


def test_bilinmeyen_operator_hata_verir():
    with pytest.raises(ValueError):
        t_norm(0.1, 0.2, "yok")


def test_siralama_ve_ozellikleri():
    # VE <= min <= max <= VEYA  (Łukasiewicz <= çarpım <= Gödel)
    assert t_norm(A, B, "lukasiewicz") <= t_norm(A, B, "product") <= t_norm(A, B, "godel")
    assert t_conorm(A, B, "godel") <= t_conorm(A, B, "product") <= t_conorm(A, B, "lukasiewicz")
