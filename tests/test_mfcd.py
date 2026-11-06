"""MFCD çekirdeği için elle hesaplanmış testler."""
import numpy as np
import pytest

from src.data_synthetic import generate
from src.memberships import membership as model_membership
from src.data_synthetic import membership as gen_membership
from src.mfcd import (Params, eta_matrix, observation_weights, predict_proba,
                      weighted_loglik)


def test_uyelik_theta_esit_beta_ise_yarim():
    a = model_membership(np.array([0.5]), np.array([1.2]), np.array([0.5]))
    assert a[0, 0] == pytest.approx(0.5)


def test_uyelik_uretec_ile_ayni():
    th, al, be = np.array([-1.0, 0.3]), np.array([0.8, 1.4]), np.array([0.1, -0.2])
    assert np.allclose(model_membership(th, al, be), gen_membership(th, al, be))


def test_bilgi_acigi_elle_hesap():
    # a = [0.8, 0.4], mu = [1, 0.5]  ->  η = 1 - max(1*0.2, 0.5*0.6) = 0.7
    a = np.array([[0.8, 0.4]])
    Q = np.array([[1, 1]])
    mu = np.array([[1.0, 0.5]])
    eta = eta_matrix(a, Q, mu, np.array([False]))
    assert eta[0, 0] == pytest.approx(0.7)


def test_ikili_mu_ile_acik_min_a_ya_iner():
    a = np.array([[0.9, 0.4, 0.7]])
    Q = np.array([[1, 1, 0]])               # 3. beceri gerekmiyor
    eta = eta_matrix(a, Q, Q.astype(float), np.array([False]))
    assert eta[0, 0] == pytest.approx(0.4)   # min(0.9, 0.4)


def test_subjektif_telafi_edici():
    a = np.array([[0.9, 0.2]])
    Q = np.array([[1, 1]])
    mu = np.array([[1.0, 1.0]])
    s = np.array([True])
    assert eta_matrix(a, Q, mu, s, or_kind="godel")[0, 0] == pytest.approx(0.9)
    assert eta_matrix(a, Q, mu, s, or_kind="product")[0, 0] == pytest.approx(0.92)


def test_operator_degisince_objektif_eta_sirasi_degisir():
    a = np.array([[0.8, 0.6]])
    Q = np.array([[1, 1]])
    mu = np.ones((1, 2))
    o = np.array([False])
    g = eta_matrix(a, Q, mu, o, and_kind="godel")[0, 0]
    p = eta_matrix(a, Q, mu, o, and_kind="product")[0, 0]
    l = eta_matrix(a, Q, mu, o, and_kind="lukasiewicz")[0, 0]
    assert (g, p, l) == pytest.approx((0.6, 0.48, 0.4))


def test_slip_guess_formulu():
    # θ = β -> a = 0.5; tek beceri, mu=1 -> η = 0.5
    p = Params(theta=np.array([0.0]), alpha=np.array([1.0]), beta=np.array([0.0]),
               slip=np.array([0.1]), guess=np.array([0.2]))
    P = predict_proba(p, np.array([[1]]), np.array([[1.0]]), np.array([False]))
    assert P[0, 0] == pytest.approx(0.9 * 0.5 + 0.2 * 0.5)   # 0.55


def test_agirliklar_elle_hesap():
    R = np.array([[1, 0]])
    dt = np.array([[45.0, 45.0]])
    c = np.array([[60.0, 120.0]])
    # max süre (tek öğrenci) = [60, 120] -> ωC = 0 ; zamanı elle test et
    w = observation_weights(R, dt, c, t_m=180.0, use_time=True)
    assert w[0] == pytest.approx([0.75, 0.75])
    # iki öğrenciyle süre: maxC = 120; C=60, doğru -> ωC = 0.5 ; w = 0.75*(1+0.5)
    R2 = np.array([[1], [1]])
    w2 = observation_weights(R2, np.array([[45.0], [45.0]]), np.array([[60.0], [120.0]]),
                             use_time=True, use_duration=True, lam=1.0)
    assert w2[0, 0] == pytest.approx(1.125)
    assert w2[1, 0] == pytest.approx(0.75)


def test_kapali_faktorler_nötr():
    R = np.ones((2, 3))
    assert (observation_weights(R) == 1.0).all()


def test_olabilirlik_ve_gercek_parametre_daha_iyi():
    # Gerçek parametreler rastgele parametrelerden daha yüksek olabilirlik vermeli.
    d = generate(seed=2)
    gercek = Params(d.theta, d.alpha, d.beta, d.slip, d.guess)
    rng = np.random.default_rng(0)
    rastgele = Params(rng.normal(size=d.theta.shape), d.alpha, d.beta, d.slip, d.guess)
    w = np.ones(d.R.shape)
    ll_g = weighted_loglik(d.R, predict_proba(gercek, d.Q, d.mu, d.is_subjective), w)
    ll_r = weighted_loglik(d.R, predict_proba(rastgele, d.Q, d.mu, d.is_subjective), w)
    assert ll_g > ll_r


def test_olasiliklar_0_1_araliginda():
    d = generate(seed=2)
    p = Params(d.theta, d.alpha, d.beta, d.slip, d.guess)
    for kind in ["godel", "product", "lukasiewicz"]:
        P = predict_proba(p, d.Q, d.mu, d.is_subjective, kind, kind)
        assert P.min() >= 0 and P.max() <= 1
