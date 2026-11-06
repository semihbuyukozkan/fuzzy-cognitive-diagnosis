"""Sentetik veri üreteci testleri."""
import numpy as np

from src.data_synthetic import (N_ITEMS, N_SKILLS, N_STUDENTS, T_M, generate,
                                membership)


def test_boyutlar_makale_yapisina_uygun():
    d = generate(seed=1)
    assert d.R.shape == (N_STUDENTS, N_ITEMS) == (1000, 50)
    assert d.Q.shape == (50, N_SKILLS) == (50, 25)
    assert (d.item_type == "reading").sum() == 38
    assert (d.item_type == "writing").sum() == 6
    assert (d.item_type == "translation").sum() == 6
    assert d.is_subjective.sum() == 12


def test_ayni_seed_ayni_veri():
    a, b = generate(seed=3), generate(seed=3)
    assert np.array_equal(a.R, b.R)
    assert np.array_equal(a.duration, b.duration)


def test_farkli_seed_farkli_veri():
    assert not np.array_equal(generate(seed=3).R, generate(seed=4).R)


def test_cevaplar_ikili_ve_q_gecerli():
    d = generate(seed=1)
    assert set(np.unique(d.R)) <= {0, 1}
    assert (d.Q.sum(axis=1) >= 1).all() and (d.Q.sum(axis=1) <= 4).all()
    assert (d.Q.sum(axis=0) >= 1).all()   # her becerinin en az bir sorusu var


def test_mu_dereceli_ve_q_disinda_sifir():
    d = generate(seed=1)
    assert (d.mu[d.Q == 0] == 0).all()
    on = d.mu[d.Q == 1]
    assert on.min() > 0.5 and on.max() <= 1.0
    assert len(np.unique(on.round(3))) > 10   # ikili değil, gerçekten dereceli


def test_zaman_ve_sure_araliklari():
    d = generate(seed=1)
    assert d.delta_t.min() >= 0 and d.delta_t.max() <= T_M
    assert d.duration.min() > 0


def test_uyelik_fonksiyonu_ozellikleri():
    # θ = β ise üyelik derecesi tam 0.5 olmalı; θ büyüdükçe artmalı.
    a = membership(np.array([0.0, 1.0, -1.0]), np.array([1.0]), np.array([0.0]))
    assert abs(a[0, 0] - 0.5) < 1e-12
    assert a[1, 0] > 0.5 > a[2, 0]
    assert ((a > 0) & (a < 1)).all()


def test_yetenek_ile_basari_pozitif_iliskili():
    d = generate(seed=1)
    r = np.corrcoef(d.theta, d.R.mean(axis=1))[0, 1]
    assert r > 0.5, f"θ ile başarı korelasyonu beklenenden düşük: {r:.2f}"


def test_dogru_orani_makul_aralikta():
    d = generate(seed=1)
    assert 0.25 < d.R.mean() < 0.75
