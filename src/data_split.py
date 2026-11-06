"""Eğitim / test bölmesi (makale Şekil 3 ile uyumlu).

Her öğrencinin kayıtlarının %p'si rastgele eğitime, kalanı teste ayrılır.
Böylece her öğrenci hem eğitimde hem testte temsil edilir (θ_i kestirilebilir).
"""
import numpy as np


def split_per_student(shape, ratio, seed):
    """Dönüş: train_mask (N, J) bool. Test = ~train_mask. Her öğrencide en az 1 eğitim ve 1 test kaydı."""
    N, J = shape
    rng = np.random.default_rng(seed)
    n_train = int(np.clip(round(ratio * J), 1, J - 1))
    mask = np.zeros(shape, dtype=bool)
    for i in range(N):
        mask[i, rng.choice(J, size=n_train, replace=False)] = True
    return mask
