"""R-hat'ın bilinen durumlarda doğru davrandığını sınar."""
import numpy as np

from src.diagnostics import ozet, split_rhat


def test_ayni_dagilimdan_zincirler_rhat_1_e_yakin():
    rng = np.random.default_rng(0)
    r = split_rhat(rng.standard_normal((4, 400, 50)))
    assert r.max() < 1.05


def test_ortalamalari_farkli_zincirler_yuksek_rhat():
    rng = np.random.default_rng(0)
    d = rng.standard_normal((4, 400, 1))
    d[0] += 3.0                                   # bir zincir başka yerde
    assert split_rhat(d)[0] > 1.1


def test_surukleyen_zincir_yakalanir():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, 400)[None, :, None]     # hepsi yukarı doğru sürükleniyor
    d = rng.standard_normal((4, 400, 1)) * 0.3 + t
    assert split_rhat(d)[0] > 1.1


def test_ozet_sayar():
    o = ozet(np.array([1.0, 1.02, 1.07, 1.2]))
    assert o["n"] == 4 and o["n_ust_1.1"] == 1 and o["n_ust_1.05"] == 2
