"""Sentetik veri üreteci (AÇIKÇA SENTETİK).

Makalenin deney yapısını taklit eder (makale verisi kapalıdır):
  - 1000 öğrenci, 50 soru: 38 okuma (objektif), 6 yazma + 6 çeviri (subjektif)
  - 25 bilişsel nitelik (bilgi noktası), her soru 1-4 niteliğe bağlı
  - her cevap için kayıt zamanı (Δt) ve cevap süresi (C)

AMAÇ: Gerçek veride sınanamayan zaman / süre / bilgi açığı faktörlerinin ve
parametre geri kazanımının (bilinen gerçek θ ile) sınanması.

UYARI (döngüsellik): Veri bu faktörlerin etkisiyle ÜRETİLDİĞİ için onları
modelleyen yöntem doğal olarak avantajlıdır. Sonuçlar faktörlerin
"geri kazanılabilirliğini" gösterir; gerçek dünya faydasını KANITLAMAZ.

Üretim hikâyesi (ground truth):
  1. Öğrencinin gizli yetenek düzeyi θ_i ~ N(0,1).
  2. Beceri k için ayırt edicilik α_k, zorluk β_k vardır.
     Yeterlilik (üyelik derecesi)  a_ik = 1 / (1 + exp(-1.7 α_k (θ - β_k)))   [Eş. 1]
  3. Zaman: öğrenci zamanla gelişir. Δt gün önce verilen cevap, bugünkünden
     daha düşük yetenekle verilmiştir:  θ(Δt) = θ - DRIFT * Δt / T_M.
     (Eski kayıtlar bugünkü duruma daha az bilgi taşır -> zaman ağırlığı fikri.)
  4. Soru başarısı η:
       objektif  (bağlayıcı, VE)   : η = 1 - max_k μ_jk (1 - a_ik)     [Eş. 8-9]
       subjektif (telafi edici, VEYA): η = max_k μ_jk a_ik
     Burada μ_jk ∈ (0.5, 1] DERECELİ üretilir (ikili olsaydı bilgi açığı
     min operatörüne indirgenirdi; bkz. docs/assumptions.md).
  5. Slip s_j ve guess g_j:  P(r=1) = (1 - s_j) η + g_j (1 - η).
  6. Cevap süresi C: doğru ve yetkin cevaplar daha hızlıdır; şans (guess)
     ile gelen doğrular çok hızlıdır (süre, cevabın güvenilirliğini taşır).
"""
from dataclasses import dataclass

import numpy as np

N_STUDENTS = 1000
N_READING, N_WRITING, N_TRANSLATION = 38, 6, 6
N_ITEMS = N_READING + N_WRITING + N_TRANSLATION  # 50
N_SKILLS = 25
T_M = 180.0      # zaman penceresi (gün)
DRIFT = 0.6      # 180 gün öncesinde yetenek 0.6 birim daha düşüktü


@dataclass
class SyntheticData:
    R: np.ndarray            # (N, J) ikili cevap
    Q: np.ndarray            # (J, K) ikili soru-beceri matrisi
    mu: np.ndarray           # (J, K) derecelendirilmiş önem (Q dışında 0)
    is_subjective: np.ndarray  # (J,) True = yazma/çeviri
    item_type: np.ndarray    # (J,) "reading" / "writing" / "translation"
    delta_t: np.ndarray      # (N, J) cevabın kaç gün önce verildiği
    duration: np.ndarray     # (N, J) cevap süresi (saniye)
    theta: np.ndarray        # (N,) gerçek yetenek (geri kazanım testi için)
    alpha: np.ndarray        # (K,) gerçek ayırt edicilik
    beta: np.ndarray         # (K,) gerçek zorluk
    slip: np.ndarray         # (J,)
    guess: np.ndarray        # (J,)
    seed: int


def membership(theta, alpha, beta):
    """Eş. 1: a_ik = 1 / (1 + exp(-1.7 α_k (θ_i - β_k))). Çıktı (N, K)."""
    z = 1.7 * alpha[None, :] * (theta[:, None] - beta[None, :])
    return 1.0 / (1.0 + np.exp(-z))


def generate(seed=0, with_effects=True):
    """Verilen seed ile tamamen tekrarlanabilir veri üretir.

    with_effects=True : zaman kayması (DRIFT) ve süre bilgisi cevabın güvenilirliğini taşır.
    with_effects=False: NEGATİF KONTROL. Zaman ve süre cevaplardan bağımsızdır (kayma yok,
                        süre yalnızca gürültü). Dereceli μ ve slip/guess korunur. Bu varyantta
                        zaman/süre faktörlü modeller baseline'dan belirgin iyiyse bu bir uyarıdır.
    """
    rng = np.random.default_rng(seed)
    N, J, K = N_STUDENTS, N_ITEMS, N_SKILLS

    item_type = np.array(["reading"] * N_READING + ["writing"] * N_WRITING
                         + ["translation"] * N_TRANSLATION)
    is_subj = item_type != "reading"

    # Q-matrisi: her soru 1-4 beceriye bağlı; her becerinin en az bir sorusu olsun.
    Q = np.zeros((J, K), dtype=int)
    for j in range(J):
        n_skill = rng.integers(1, 5)
        Q[j, rng.choice(K, size=n_skill, replace=False)] = 1
    for k in range(K):
        if Q[:, k].sum() == 0:
            Q[rng.integers(0, J), k] = 1

    # Dereceli önem μ_jk: yalnızca Q'nun 1 olduğu yerlerde (0.5, 1].
    mu = Q * rng.uniform(0.5, 1.0, size=(J, K))

    # Parametreler
    theta = rng.normal(0.0, 1.0, size=N)
    alpha = rng.uniform(0.5, 1.5, size=K)
    beta = rng.normal(0.0, 0.7, size=K)
    slip = rng.uniform(0.02, 0.15, size=J)
    guess = rng.uniform(0.02, 0.25, size=J)

    drift = DRIFT if with_effects else 0.0
    # Cevap zamanları: kaç gün önce verildi
    delta_t = rng.uniform(0.0, T_M, size=(N, J))

    # η hesabı: her (öğrenci, soru) için, o andaki yetenekle
    eta = np.zeros((N, J))
    for j in range(J):
        theta_t = theta - drift * delta_t[:, j] / T_M            # (N,)
        a = membership(theta_t, alpha, beta)                      # (N, K)
        if is_subj[j]:
            eta[:, j] = (mu[j][None, :] * a).max(axis=1)          # VEYA
        else:
            eta[:, j] = 1.0 - (mu[j][None, :] * (1.0 - a)).max(axis=1)  # VE (açık)

    # Slip / guess ile gözlenen doğru olasılığı ve cevap
    p = (1.0 - slip[None, :]) * eta + guess[None, :] * (1.0 - eta)
    R = (rng.random((N, J)) < p).astype(int)

    # Cevap süresi (saniye): log-normal; yetkin ve doğru -> hızlı, şans -> çok hızlı
    base = rng.uniform(40, 200, size=J)                            # sorunun tipik süresi
    speed = np.exp(-0.5 * (eta - 0.5)) if with_effects else 1.0
    duration = base[None, :] * speed * np.exp(rng.normal(0, 0.25, size=(N, J)))
    # Şansla doğru bilenler (η düşükken doğru çıkanlar) çok hızlı cevaplar
    lucky = (R == 1) & (rng.random((N, J)) < (1.0 - eta) * guess[None, :] / np.maximum(p, 1e-9))
    if with_effects:
        duration = np.where(lucky, duration * 0.3, duration)

    return SyntheticData(R=R, Q=Q, mu=mu, is_subjective=is_subj, item_type=item_type,
                         delta_t=delta_t, duration=duration, theta=theta, alpha=alpha,
                         beta=beta, slip=slip, guess=guess, seed=seed)


if __name__ == "__main__":
    d = generate(seed=0)
    print(f"R: {d.R.shape}, doğru oranı: {d.R.mean():.3f}")
    print(f"Objektif doğru oranı: {d.R[:, ~d.is_subjective].mean():.3f}, "
          f"subjektif: {d.R[:, d.is_subjective].mean():.3f}")
    print(f"Soru başına beceri (min/maks): {d.Q.sum(1).min()}/{d.Q.sum(1).max()}")
    print(f"Süre ortalaması: {d.duration.mean():.1f} sn")
