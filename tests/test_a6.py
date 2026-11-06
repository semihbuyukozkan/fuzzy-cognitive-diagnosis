"""A6 testleri: min'e indirgenme, zaman çarpanı, Łukasiewicz uç değerleri, negatif kontrol bayrağı."""
import numpy as np
import pytest

from src.data_synthetic import generate
from src.mfcd import Params, eta_matrix, predict_proba, weighted_loglik


def _rastgele(seed, N=200, J=15, K=6):
    rng = np.random.default_rng(seed)
    a = rng.random((N, K))
    Q = (rng.random((J, K)) < 0.4).astype(int)
    Q[np.arange(J), rng.integers(0, K, J)] = 1          # her soruda en az 1 beceri
    return rng, a, Q


def _eq89_dongu(a, Q, mu):
    """Makalenin Eş. 8-9'u kelimesi kelimesine: l_ijk = μ_jk (1 - a_ik); η = 1 - max_k l_ijk."""
    N, K = a.shape
    J = Q.shape[0]
    eta = np.zeros((N, J))
    for i in range(N):
        for j in range(J):
            gap = [mu[j, k] * (1 - a[i, k]) for k in range(K) if Q[j, k] == 1]
            eta[i, j] = 1 - max(gap)
    return eta


@pytest.mark.parametrize("seed", range(5))
def test_ikili_mu_ile_bilgi_acigi_tam_min_a_dir(seed):
    """İkili μ = Q iken Eş. 8-9, Gödel t-norm'una (min) BİREBİR eşittir."""
    _, a, Q = _rastgele(seed)
    eta = eta_matrix(a, Q, Q.astype(float), np.zeros(len(Q), bool), "godel")
    min_a = np.array([[a[i, Q[j] == 1].min() for j in range(len(Q))] for i in range(len(a))])
    assert np.allclose(eta, min_a, atol=1e-12, rtol=0)
    assert np.allclose(eta, _eq89_dongu(a, Q, Q.astype(float)), atol=1e-12, rtol=0)


@pytest.mark.parametrize("seed", range(5))
def test_dereceli_mu_ile_kod_eq89_ile_ayni(seed):
    """Dereceli μ'de de kodumuz Eş. 8-9 ile aynıdır; ama artık min(a)'dan FARKLIDIR."""
    rng, a, Q = _rastgele(seed)
    mu = Q * rng.uniform(0.5, 1.0, Q.shape)
    eta = eta_matrix(a, Q, mu, np.zeros(len(Q), bool), "godel")
    assert np.allclose(eta, _eq89_dongu(a, Q, mu), atol=1e-12, rtol=0)
    min_a = np.array([[a[i, Q[j] == 1].min() for j in range(len(Q))] for i in range(len(a))])
    assert not np.allclose(eta, min_a)


def test_zaman_carpani_elle_hesap_ve_nötr():
    a = np.array([[0.8, 0.4]])
    Q = np.array([[1, 1]])
    mu = np.array([[1.0, 0.5]])
    s = np.array([False])
    # ω = 1 -> değişiklik yok
    assert eta_matrix(a, Q, mu, s, omega=np.ones((1, 1)))[0, 0] == pytest.approx(0.7)
    # ω = 0.5 -> a_eff = [0.4, 0.2]; η = 1 - max(1*0.6, 0.5*0.8) = 0.4
    assert eta_matrix(a, Q, mu, s, omega=np.full((1, 1), 0.5))[0, 0] == pytest.approx(0.4)


def test_lukasiewicz_uc_degerlerinde_log_olabilirlik_sonlu():
    """a = 0 -> Łukasiewicz η tam 0; slip/guess sayesinde P ∈ (0,1) kalır, log(0) oluşmaz."""
    d = generate(seed=0)
    p = Params(theta=np.full(d.R.shape[0], -50.0), alpha=d.alpha, beta=d.beta,
               slip=d.slip, guess=d.guess)
    P = predict_proba(p, d.Q, d.mu, d.is_subjective, "lukasiewicz", "lukasiewicz")
    assert P.min() > 0 and P.max() < 1
    ll = weighted_loglik(d.R, P, np.ones(d.R.shape))
    assert np.isfinite(ll)
    p2 = Params(theta=np.full(d.R.shape[0], 50.0), alpha=d.alpha, beta=d.beta,
                slip=d.slip, guess=d.guess)
    P2 = predict_proba(p2, d.Q, d.mu, d.is_subjective, "lukasiewicz", "lukasiewicz")
    assert P2.max() < 1 and np.isfinite(weighted_loglik(d.R, P2, np.ones(d.R.shape)))


def _sure_korelasyonu(d):
    ld = np.log(d.duration)
    ld = ld - ld.mean(axis=0, keepdims=True)           # soru etkisini çıkar
    return np.corrcoef(ld.ravel(), d.R.ravel())[0, 1]


def test_negatif_kontrol_zaman_ve_sure_cevaptan_bagimsiz():
    on, off = generate(seed=1, with_effects=True), generate(seed=1, with_effects=False)
    # etkili varyantta: eski kayıtlar daha az doğru ve hızlı cevaplar daha çok doğru
    eski = on.delta_t > 90
    assert on.R[eski].mean() < on.R[~eski].mean() - 0.01
    assert _sure_korelasyonu(on) < -0.05
    # etkisiz varyantta: zaman ve süre cevapla ilişkisiz (yalnızca θ-etkisi yok)
    eski = off.delta_t > 90
    assert abs(off.R[eski].mean() - off.R[~eski].mean()) < 0.015
    assert abs(_sure_korelasyonu(off)) < 0.03


def test_negatif_kontrolde_mu_dereceli_kalir_ve_ayni_q():
    on, off = generate(seed=1, with_effects=True), generate(seed=1, with_effects=False)
    assert np.array_equal(on.Q, off.Q) and np.allclose(on.mu, off.mu)
    assert len(np.unique(off.mu[off.Q == 1].round(3))) > 10
