"""MH kestirimi testleri (kısa zincirlerle; tam yakınsama deneyleri experiments/ altında)."""
import numpy as np

from src.data_synthetic import generate
from src.mcmc import MCMCConfig, run_mh


def _kur(seed=0, oran=0.6):
    d = generate(seed=seed)
    train = np.random.default_rng(seed).random(d.R.shape) < oran
    return d, train, np.ones(d.R.shape)


def test_ayni_seed_ayni_sonuc():
    d, train, w = _kur()
    cfg = MCMCConfig(n_iter=20, burn=10, thin=5, seed=7)
    a = run_mh(d.R, d.Q, d.is_subjective, train, w, cfg)
    b = run_mh(d.R, d.Q, d.is_subjective, train, w, cfg)
    assert np.array_equal(a["P_mean"], b["P_mean"])


def test_cikti_gecerli_ve_kabul_oranlari_makul():
    d, train, w = _kur()
    out = run_mh(d.R, d.Q, d.is_subjective, train, w, MCMCConfig(n_iter=60, burn=30, thin=5))
    assert out["P_mean"].shape == d.R.shape
    assert out["P_mean"].min() >= 0 and out["P_mean"].max() <= 1
    for ad, oran in out["accept"].items():
        assert 0.05 < oran < 0.95, f"{ad} kabul oranı uç değerde: {oran:.2f}"


def test_olabilirlik_artar_ve_theta_geri_kazanilir():
    d, train, w = _kur()
    out = run_mh(d.R, d.Q, d.is_subjective, train, w,
                 MCMCConfig(n_iter=200, burn=100, thin=5))
    tr = out["ll_trace"]
    assert tr[-50:].mean() > tr[:5].mean()          # model veriyi giderek daha iyi açıklıyor
    r = np.corrcoef(out["theta_mean"], d.theta)[0, 1]
    assert r > 0.75, f"θ geri kazanımı zayıf: {r:.2f}"


def test_mu_kapaliyken_mu_q_ye_esit_kalir():
    d, train, w = _kur()
    out = run_mh(d.R, d.Q, d.is_subjective, train, w,
                 MCMCConfig(n_iter=20, burn=10, thin=5, learn_mu=False))
    assert np.array_equal(out["mu"], d.Q.astype(float))
