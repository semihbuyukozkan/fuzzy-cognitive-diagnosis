"""Baseline modellerin akıl sağlığı testleri."""
import numpy as np

from src.baselines import fit_dina, fit_irt2pl
from src.data_split import split_per_student
from src.data_synthetic import generate


def _veri():
    d = generate(seed=1)
    train = split_per_student(d.R.shape, 0.6, seed=1)
    return d, train


def test_irt_olasiliklari_gecerli_ve_cogunluktan_iyi():
    d, train = _veri()
    P = fit_irt2pl(d.R, train)
    assert P.min() >= 0 and P.max() <= 1
    te = ~train
    dogruluk = ((P[te] >= 0.5) == d.R[te]).mean()
    cogunluk = max(d.R[te].mean(), 1 - d.R[te].mean())
    assert dogruluk > cogunluk + 0.05


def test_dina_gecerli_ve_tekrarlanabilir():
    d, train = _veri()
    a = fit_dina(d.R, d.Q, train, n_iter=30, burn=10, thin=5, seed=2)
    b = fit_dina(d.R, d.Q, train, n_iter=30, burn=10, thin=5, seed=2)
    assert a.min() >= 0 and a.max() <= 1
    assert np.array_equal(a, b)
    # slip/guess <= 0.4 olduğundan olasılıklar [0, 1] içinde ve boş değil
    assert a.std() > 0.01
