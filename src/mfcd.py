"""MFCD-core: geliştirilmiş bulanık bilişsel tanı modelinin çekirdeği.

Akış (her öğrenci i, soru j için):
  1. Yeterlilik         a_ik  = üyelik fonksiyonu(θ_i, α_k, β_k)            [Eş. 1]
  2. Beceri katkısı     ã_ijk : sorunun gerektirdiği her beceri için bir derece
        objektif  : ã = 1 - μ_jk (1 - a_ik)   (eksik bilgi "açığı" cezalandırılır)
        subjektif : ã = μ_jk · a_ik
     μ_jk ∈ [0,1] becerinin o soru için önemidir. Gerekmeyen becerilerde (Q=0) ã
     "nötr" değerdir (VE için 1, VEYA için 0), yani sonucu etkilemez.
  3. Birleştirme        η_ij  = VE_k ã_ijk (objektif)  veya  VEYA_k ã_ijk (subjektif)
        VE / VEYA operatörü parametredir (godel / product / lukasiewicz)  [Eş. 3,5,6]
        Gödel (min) + objektif tam olarak bilgi açığı formülüdür:
        η = 1 - max_k μ_jk (1 - a_ik)                                     [Eş. 8-9]
  4. Slip / guess       P(r_ij = 1) = (1 - s_j) η_ij + g_j (1 - η_ij)

  Slip s_j : biliyor ama yanlış yapma (dikkatsizlik) olasılığı
  Guess g_j: bilmiyor ama doğru tahmin etme olasılığı

Not: Q ikili ve μ = Q alınırsa açık faktörü kapanmış olur (orijinal FuzzyCDF mantığı).
Bu yüzden "açık faktörü" = dereceli μ kullanmak demektir (bkz. docs/assumptions.md).

Zaman / süre ağırlıkları gözlemin olabilirlik ağırlığı olarak kullanılır
(docs/assumptions.md, madde 1-2): w_ij = ω_time · (1 + λ · ωC).
"""
from dataclasses import dataclass

import numpy as np

from .fuzzy_ops import and_reduce, or_reduce
from .memberships import membership

EPS = 1e-9


@dataclass
class Params:
    theta: np.ndarray   # (N,)
    alpha: np.ndarray   # (K,)
    beta: np.ndarray    # (K,)
    slip: np.ndarray    # (J,)
    guess: np.ndarray   # (J,)


def eta_matrix(a, Q, mu, is_subj, and_kind="godel", or_kind="godel"):
    """Soru başarı derecesi η (N, J). a: (N,K), Q/mu: (J,K), is_subj: (J,) bool."""
    on = Q[None, :, :] == 1                                     # (1, J, K)
    a3 = a[:, None, :]                                          # (N, 1, K)
    # Objektif: ilgisiz beceri -> 1 (VE için nötr)
    obj = np.where(on, 1.0 - mu[None] * (1.0 - a3), 1.0)        # (N, J, K)
    # Subjektif: ilgisiz beceri -> 0 (VEYA için nötr)
    sub = np.where(on, mu[None] * a3, 0.0)
    eta = np.empty(obj.shape[:2])
    if (~is_subj).any():
        eta[:, ~is_subj] = and_reduce(obj[:, ~is_subj, :], and_kind, axis=-1)
    if is_subj.any():
        eta[:, is_subj] = or_reduce(sub[:, is_subj, :], or_kind, axis=-1)
    return eta


def predict_proba(p, Q, mu, is_subj, and_kind="godel", or_kind="godel"):
    """P(r_ij = 1) = (1 - s_j) η + g_j (1 - η), boyut (N, J)."""
    a = membership(p.theta, p.alpha, p.beta)
    eta = eta_matrix(a, Q, mu, is_subj, and_kind, or_kind)
    return (1.0 - p.slip[None, :]) * eta + p.guess[None, :] * (1.0 - eta)


def observation_weights(R, delta_t=None, duration=None, t_m=180.0, lam=1.0,
                        use_time=False, use_duration=False):
    """Gözlem ağırlıkları w (N, J). Kapalı faktör nötr (1) davranır.

    Zaman  : ω_time = (T_m - Δt) / T_m                      [Eş. 2,4] -> eski kayıt az ağırlıklı
    Süre   : ωC = (maxC_j - C_ij) / maxC_j · R_ij            [Eş. 7, [0,1]'e ölçekli: BİZİM yorum]
             (hızlı + doğru cevap güvenilir sayılır; yanlış cevapta ωC = 0)
    Birleşik: w = ω_time · (1 + λ ωC)
    """
    w = np.ones(R.shape)
    if use_time:
        w = w * np.clip((t_m - delta_t) / t_m, 0.0, 1.0)
    if use_duration:
        max_c = duration.max(axis=0, keepdims=True)
        omega_c = np.clip((max_c - duration) / max_c, 0.0, 1.0) * R
        w = w * (1.0 + lam * omega_c)
    return w


def weighted_loglik(R, P, w, mask=None):
    """Ağırlıklı Bernoulli log-olabilirliği (Eş. 12-13'ün düzeltilmiş hali).

    mask: True olan hücreler (eğitim kayıtları) hesaba katılır.
    """
    P = np.clip(P, EPS, 1.0 - EPS)
    ll = R * np.log(P) + (1 - R) * np.log(1.0 - P)
    if mask is not None:
        ll = np.where(mask, ll, 0.0)
        w = np.where(mask, w, 0.0)
    return float((w * ll).sum())
