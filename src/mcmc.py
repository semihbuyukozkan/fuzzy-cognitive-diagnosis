"""Metropolis-Hastings (random-walk) ile MFCD parametre kestirimi (Eş. 14-16 mantığı).

Fikir: Bilmediğimiz parametreleri (θ, α, β, slip, guess, μ) bir "posterior" dağılımdan
örneklemek. Posterior ∝ olabilirlik x önsel dağılım. Her adımda:
  1. Bir parametreye küçük rastgele bir sıçrama öner:  x' = x + adım · N(0,1)
  2. Kabul olasılığı:  min(1, posterior(x') / posterior(x))        [log uzayında hesaplanır]
  3. u ~ U(0,1) çek; log u < log(oran) ise x'i kabul et, değilse x'te kal.
Öneri dağılımı simetrik olduğu için oranda öneri terimi sadeleşir. Daha iyi açıklayan
değerler hep kabul edilir; daha kötüler bazen kabul edilir (bu yüzden yerel tepelerde
sıkışmaz). Yeterince adımdan sonra örnekler posterior'a yaklaşır (burn-in atılır).

Hız için: θ_i'ler (öğrenciler) ve s_j, g_j, μ_j (sorular) birbirinden koşullu bağımsız
olduğundan hepsi bir seferde (vektörel) önerilir, her biri ayrı ayrı kabul/ret edilir.
α_k, β_k ise birden çok soruyu etkilediği için becerilerle tek tek güncellenir; yalnızca o
beceriyi gerektiren sorular yeniden hesaplanır.

Önsel dağılımlar (bizim seçimimiz): θ ~ N(0,1), β ~ N(0,1), log α ~ N(0, 0.5²),
s, g ~ U(0, 0.4), μ ~ U(0, 1]. Olabilirlik: ağırlıklı Bernoulli (standart form).
Sadece eğitim kayıtları (train_mask) olabilirliğe girer.
"""
from dataclasses import dataclass

import numpy as np

from .memberships import membership
from .mfcd import EPS, Params, eta_matrix


@dataclass
class MCMCConfig:
    n_iter: int = 1500
    burn: int = 700
    thin: int = 10
    step_theta: float = 0.9
    step_ab: float = 0.25
    step_sg: float = 0.04
    step_mu: float = 0.15
    and_kind: str = "godel"
    or_kind: str = "godel"
    learn_mu: bool = True      # False -> μ = Q (bilgi açığı kapalı, orijinal FuzzyCDF mantığı)
    seed: int = 0
    overdispersed_init: bool = False   # True: zincirler birbirinden farklı noktalardan başlar (R-hat için)
    store_samples: bool = False        # True: seyreltilmiş parametre örneklerini de döndür


def _cell_ll(R, P, w, mask):
    """Hücre başına ağırlıklı log-olabilirlik (N, J)."""
    P = np.clip(P, EPS, 1.0 - EPS)
    return np.where(mask, w * (R * np.log(P) + (1 - R) * np.log(1.0 - P)), 0.0)


def _prob(eta, slip, guess):
    return (1.0 - slip[None, :]) * eta + guess[None, :] * (1.0 - eta)


def _init_theta(R, train_mask):
    """θ için başlangıç: eğitimdeki başarı oranının z-skoru."""
    score = (R * train_mask).sum(1) / np.maximum(train_mask.sum(1), 1)
    return (score - score.mean()) / (score.std() + 1e-9)


def run_mh(R, Q, is_subj, train_mask, w, cfg=MCMCConfig(), omega=None):
    """MH çalıştırır. Dönüş: sözlük (P_mean, theta_mean, params, kabul oranları, ll izi).

    omega: (N, J) zaman çarpanı; verilirse η hesabında yeterlilikle çarpılır (Eş. 3 yazımı).
    """
    rng = np.random.default_rng(cfg.seed)
    N, J = R.shape
    K = Q.shape[1]
    Qf = Q.astype(float)

    # --- başlangıç durumu ---
    theta = _init_theta(R, train_mask)
    alpha, beta = np.ones(K), np.zeros(K)
    slip, guess = np.full(J, 0.1), np.full(J, 0.1)
    mu = Qf.copy() if not cfg.learn_mu else 0.9 * Qf
    if cfg.overdispersed_init:   # farklı zincirler farklı noktalardan başlasın
        theta = theta + 0.7 * rng.standard_normal(N)
        alpha = np.exp(0.4 * rng.standard_normal(K))
        beta = 0.7 * rng.standard_normal(K)
        slip, guess = rng.uniform(0.03, 0.3, J), rng.uniform(0.03, 0.3, J)
        if cfg.learn_mu:
            mu = Qf * rng.uniform(0.4, 1.0, (J, K))
    a = membership(theta, alpha, beta)
    eta = eta_matrix(a, Q, mu, is_subj, cfg.and_kind, cfg.or_kind)
    ll = _cell_ll(R, _prob(eta, slip, guess), w, train_mask)

    acc = {"theta": [0, 0], "alpha_beta": [0, 0], "slip_guess": [0, 0], "mu": [0, 0]}
    P_sum, n_kept, theta_sum, trace = np.zeros((N, J)), 0, np.zeros(N), []
    S = {k: [] for k in ("theta", "alpha", "beta", "slip", "guess", "mu", "ll")}

    def eta_of(a_, items=slice(None), mu_=None):
        m = mu if mu_ is None else mu_
        om = None if omega is None else omega[:, items]
        return eta_matrix(a_, Q[items], m[items], is_subj[items], cfg.and_kind, cfg.or_kind, om)

    for it in range(cfg.n_iter):
        # 1) θ: tüm öğrenciler aynı anda önerilir, ayrı ayrı kabul edilir
        th_new = theta + cfg.step_theta * rng.standard_normal(N)
        a_new = membership(th_new, alpha, beta)
        eta_new = eta_of(a_new)
        ll_new = _cell_ll(R, _prob(eta_new, slip, guess), w, train_mask)
        log_r = (ll_new.sum(1) - ll.sum(1)) - 0.5 * (th_new**2 - theta**2)
        ok = np.log(rng.random(N)) < log_r
        theta[ok], a[ok], eta[ok], ll[ok] = th_new[ok], a_new[ok], eta_new[ok], ll_new[ok]
        acc["theta"][0] += ok.sum(); acc["theta"][1] += N

        # 2) α_k, β_k: beceri beceri; yalnızca o beceriyi içeren sorular yeniden hesaplanır
        for k in rng.permutation(K):
            items = np.where(Q[:, k] == 1)[0]
            for which in ("alpha", "beta"):
                al_k, be_k = alpha[k], beta[k]
                if which == "alpha":
                    al_k = al_k + cfg.step_ab * rng.standard_normal()
                    if al_k <= 0:
                        acc["alpha_beta"][1] += 1
                        continue
                    lp = -0.5 * ((np.log(al_k) / 0.5) ** 2 - (np.log(alpha[k]) / 0.5) ** 2) \
                        - np.log(al_k) + np.log(alpha[k])      # log-normal yoğunluk (α üzerinde)
                else:
                    be_k = be_k + cfg.step_ab * rng.standard_normal()
                    lp = -0.5 * (be_k**2 - beta[k] ** 2)
                a_try = a.copy()
                a_try[:, k] = membership(theta, np.array([al_k]), np.array([be_k]))[:, 0]
                eta_try = eta_of(a_try, items)
                ll_try = _cell_ll(R[:, items], _prob(eta_try, slip[items], guess[items]),
                                  w[:, items], train_mask[:, items])
                if np.log(rng.random()) < (ll_try.sum() - ll[:, items].sum()) + lp:
                    alpha[k], beta[k], a = al_k, be_k, a_try
                    eta[:, items], ll[:, items] = eta_try, ll_try
                    acc["alpha_beta"][0] += 1
                acc["alpha_beta"][1] += 1

        # 3) slip, guess: soru başına (eta sabit)
        s_new = slip + cfg.step_sg * rng.standard_normal(J)
        g_new = guess + cfg.step_sg * rng.standard_normal(J)
        valid = (s_new > 0) & (s_new < 0.4) & (g_new > 0) & (g_new < 0.4)
        ll_new = _cell_ll(R, _prob(eta, np.clip(s_new, 0, 0.4), np.clip(g_new, 0, 0.4)), w, train_mask)
        ok = valid & (np.log(rng.random(J)) < (ll_new.sum(0) - ll.sum(0)))
        slip[ok], guess[ok], ll[:, ok] = s_new[ok], g_new[ok], ll_new[:, ok]
        acc["slip_guess"][0] += ok.sum(); acc["slip_guess"][1] += J

        # 4) μ: sorunun gerektirdiği becerilerin önemi birlikte önerilir
        if cfg.learn_mu:
            mu_new = mu + cfg.step_mu * rng.standard_normal((J, K)) * Qf
            valid = ((mu_new * Qf > 0) | (Q == 0)).all(1) & (mu_new <= 1.0).all(1)
            eta_new = eta_of(a, mu_=mu_new)
            ll_new = _cell_ll(R, _prob(eta_new, slip, guess), w, train_mask)
            ok = valid & (np.log(rng.random(J)) < (ll_new.sum(0) - ll.sum(0)))
            mu[ok], eta[:, ok], ll[:, ok] = mu_new[ok], eta_new[:, ok], ll_new[:, ok]
            acc["mu"][0] += ok.sum(); acc["mu"][1] += J

        trace.append(ll.sum())
        if it >= cfg.burn and (it - cfg.burn) % cfg.thin == 0:
            P_sum += _prob(eta, slip, guess)
            theta_sum += theta
            n_kept += 1
            if cfg.store_samples:
                for k_, v_ in (("theta", theta), ("alpha", alpha), ("beta", beta), ("slip", slip),
                               ("guess", guess), ("mu", mu[Q == 1]), ("ll", np.array([ll.sum()]))):
                    S[k_].append(v_.copy())

    post = np.array(trace[cfg.burn:])
    yarim = len(post) // 2
    # Sürüklenme ölçüsü: burn-in sonrası iznin ilk yarısı ile ikinci yarısının ortalama farkı,
    # iz standart sapması cinsinden. 0'a yakınsa iz düzleşmiştir; büyükse hâlâ yükseliyor/iniyor.
    drift = float(abs(post[:yarim].mean() - post[yarim:].mean()) / (post.std() + 1e-12))
    return {
        "ll_drift": drift,
        "samples": {k_: np.array(v_) for k_, v_ in S.items()} if cfg.store_samples else None,
        "P_mean": P_sum / n_kept,               # posterior ortalama cevap olasılığı
        "theta_mean": theta_sum / n_kept,
        "params": Params(theta, alpha, beta, slip, guess),
        "mu": mu,
        "accept": {k: v[0] / max(v[1], 1) for k, v in acc.items()},
        "ll_trace": np.array(trace),
        "n_samples": n_kept,
    }
