"""Baseline modeller (kendi gerçekleştirmemiz; harici kod kullanılmadı).

1) IRT-2PL  : klasik madde tepki kuramı.  P(r_ij=1) = sigmoid(a_j (θ_i - b_j)).
              Parametreler MAP (olabilirlik + zayıf önsel) ile L-BFGS kullanılarak bulunur.
2) DINA     : ayrık (0/1) beceri modeli. Öğrenci bir beceriyi ya biliyor ya bilmiyor;
              soru için gereken TÜM becerileri biliyorsa η=1, yoksa 0.
              P(r=1) = (1-s_j) η + g_j (1-η).  Beceri durumları Gibbs örnekleyici ile
              kestirilir (2^25 örüntüyü taramak mümkün olmadığı için EM yerine).
3) FuzzyCDF-orijinal : mcmc.run_mh(learn_mu=False, godel) ve ağırlıksız
              (üç faktör kapalı, VE = min, VEYA = max). experiment.py içinde çağrılır.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .mfcd import EPS


# ----------------------------------------------------------------------------- IRT-2PL
def fit_irt2pl(R, train_mask, seed=0):
    """MAP ile IRT-2PL. Dönüş: P (N, J) tahmin olasılıkları."""
    N, J = R.shape
    M = train_mask.astype(float)
    rng = np.random.default_rng(seed)
    x0 = np.concatenate([0.1 * rng.standard_normal(N), np.zeros(J), np.zeros(J)])  # θ, log a, b

    def unpack(x):
        return x[:N], x[N:N + J], x[N + J:]

    def negpost(x):
        th, la, b = unpack(x)
        a = np.exp(la)
        z = a[None, :] * (th[:, None] - b[None, :])
        p = np.clip(expit(z), EPS, 1 - EPS)
        ll = (M * (R * np.log(p) + (1 - R) * np.log(1 - p))).sum()
        prior = -0.5 * (th**2).sum() - 0.5 * (b**2).sum() - 0.5 * ((la / 0.5) ** 2).sum()
        g = M * (R - p)                                   # dℓ/dz
        d_th = (g * a[None, :]).sum(1) - th
        d_la = (g * (th[:, None] - b[None, :])).sum(0) * a - la / 0.25
        d_b = -(g * a[None, :]).sum(0) - b
        return -(ll + prior), -np.concatenate([d_th, d_la, d_b])

    res = minimize(negpost, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    th, la, b = unpack(res.x)
    return expit(np.exp(la)[None, :] * (th[:, None] - b[None, :]))


# ----------------------------------------------------------------------------- DINA
def fit_dina(R, Q, train_mask, n_iter=1000, burn=500, thin=5, seed=0):
    """DINA, Gibbs örnekleyici. Dönüş: posterior ortalama P (N, J).

    Her adımda: (1) her becerinin durumu (0/1) tüm öğrenciler için koşullu olasılığından
    örneklenir, (2) s_j ve g_j Beta dağılımından (U(0,0.4) önsel, kesilmiş) örneklenir.
    """
    rng = np.random.default_rng(seed)
    N, J = R.shape
    K = Q.shape[1]
    Qf = Q.astype(float)
    alpha = (rng.random((N, K)) < 0.5).astype(float)
    slip, guess = np.full(J, 0.1), np.full(J, 0.1)
    miss = (1.0 - alpha) @ Qf.T                      # (N,J): eksik olan gerekli beceri sayısı

    def cell_ll(eta_, idx):
        p = np.clip((1 - slip[idx])[None, :] * eta_ + guess[idx][None, :] * (1 - eta_), EPS, 1 - EPS)
        r = R[:, idx]
        return np.where(train_mask[:, idx], r * np.log(p) + (1 - r) * np.log(1 - p), 0.0)

    P_sum, kept = np.zeros((N, J)), 0
    for it in range(n_iter):
        for k in rng.permutation(K):
            idx = np.where(Q[:, k] == 1)[0]
            qk = Qf[idx][None, :, k]                              # (1, |idx|)
            miss_if0 = miss[:, idx] + alpha[:, [k]] * qk           # k'yı bilmiyor durumu
            miss_if1 = miss[:, idx] - (1 - alpha[:, [k]]) * qk     # k'yı biliyor durumu
            ll0 = cell_ll((miss_if0 == 0).astype(float), idx).sum(1)
            ll1 = cell_ll((miss_if1 == 0).astype(float), idx).sum(1)
            p1 = 1.0 / (1.0 + np.exp(np.clip(ll0 - ll1, -50, 50)))   # önsel 0.5/0.5
            new = (rng.random(N) < p1).astype(float)
            miss[:, idx] = np.where(new[:, None] == 1, miss_if1, miss_if0)
            alpha[:, k] = new
        eta = (miss == 0).astype(float)
        # s_j: η=1 olanlardan yanlış yapanlar; g_j: η=0 olanlardan doğru yapanlar
        m = train_mask
        n_s_bad = (eta * (1 - R) * m).sum(0); n_s_ok = (eta * R * m).sum(0)
        n_g_ok = ((1 - eta) * R * m).sum(0); n_g_bad = ((1 - eta) * (1 - R) * m).sum(0)
        slip = _trunc_beta(rng, 1 + n_s_bad, 1 + n_s_ok)
        guess = _trunc_beta(rng, 1 + n_g_ok, 1 + n_g_bad)
        if it >= burn and (it - burn) % thin == 0:
            P_sum += (1 - slip)[None, :] * eta + guess[None, :] * (1 - eta)
            kept += 1
    return P_sum / kept


def _trunc_beta(rng, a, b, upper=0.4):
    """Beta(a,b) örneği, (0, upper) ile sınırlı (reddetme yöntemiyle; olmazsa kırpılır)."""
    x = rng.beta(a, b)
    for _ in range(50):
        bad = x >= upper
        if not bad.any():
            break
        x[bad] = rng.beta(a[bad], b[bad])
    return np.clip(x, 1e-3, upper - 1e-3)
