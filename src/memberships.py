"""Üyelik fonksiyonu (makale Eş. 1).

Bulanık küme: bir elemanın kümeye aitliği 0/1 değil, 0..1 arası bir dereceyle ifade edilir.
Burada küme "beceri k'yı yeterince bilen öğrenciler" kümesidir ve öğrenci i'nin
üyelik derecesi onun beceri k'daki YETERLİLİĞİDİR:

    a_ik = μ_k(i) = 1 / (1 + exp(-1.7 · α_k · (θ_i - β_k)))

  θ_i : öğrencinin genel yetenek düzeyi
  β_k : becerinin zorluğu (θ = β ise üyelik tam 0.5)
  α_k : ayırt edicilik (eğrinin dikliği; büyükse geçiş keskin)
  1.7 : lojistik eğriyi normal ogive'e yaklaştıran klasik sabit

Gösterim notu: makale hem ayırt edicilik hem yeterlilik için "a" kullanıyor;
biz α (ayırt edicilik), β (zorluk), a (yeterlilik) kullanıyoruz.
"""
import numpy as np

D = 1.7


def membership(theta, alpha, beta):
    """theta (N,), alpha (K,), beta (K,) -> yeterlilik matrisi a (N, K)."""
    z = D * alpha[None, :] * (theta[:, None] - beta[None, :])
    return 1.0 / (1.0 + np.exp(-z))
