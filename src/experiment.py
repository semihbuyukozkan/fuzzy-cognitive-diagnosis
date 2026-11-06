"""Deney koşucusu: veri yükleme, model çağırma, değerlendirme, paralel çalıştırma.

Modeller (isim -> açıklama):
  IRT-2PL         : baseline
  DINA            : baseline
  FuzzyCDF-orig   : MFCD çekirdeği, μ=Q (açık kapalı), min/max, ağırlıksız
  MFCD-core       : MFCD çekirdeği, μ öğrenilir (açık açık), min/max, ağırlıksız (varsayılan)
  Ayrıca doğrudan `mfcd:<ayar>` biçiminde ayarlanabilir (A6: operatör ve ablation).
"""
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .baselines import fit_dina, fit_irt2pl
from .data_frcsub import load_frcsub
from .data_split import split_per_student
from .data_synthetic import generate
from .mcmc import MCMCConfig, run_mh
from .metrics import evaluate
from .mfcd import observation_weights

SEEDS = list(range(10))                       # sabit ve kayıtlı seed'ler
RATIOS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class MfcdSpec:
    """MFCD çalıştırma ayarı (K1 ve K2 için)."""
    learn_mu: bool = False        # açık faktörü (dereceli μ)
    use_time: bool = False        # zaman ağırlığı
    use_duration: bool = False    # süre ağırlığı
    and_kind: str = "godel"
    or_kind: str = "godel"
    n_iter: int = 1500
    burn: int = 700


STANDART = {
    "FuzzyCDF-orig": MfcdSpec(learn_mu=False),
    "MFCD-core": MfcdSpec(learn_mu=True),
}


def load_data(name, seed=0):
    """Dönüş: sözlük (R, Q, is_subj, delta_t, duration)."""
    if name == "frcsub":
        R, Q = load_frcsub()
        return dict(R=R, Q=Q, is_subj=np.zeros(R.shape[1], bool), delta_t=None, duration=None)
    d = generate(seed=seed)       # sentetik: veri seed'i = deney seed'i
    return dict(R=d.R, Q=d.Q, is_subj=d.is_subjective, delta_t=d.delta_t, duration=d.duration)


def run_one(data_name, model, ratio, seed, spec=None):
    """Tek koşu: bölme + eğitim + test metrikleri + süre. Dönüş: sözlük."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    D = load_data(data_name, seed)
    R, Q, subj = D["R"], D["Q"], D["is_subj"]
    train = split_per_student(R.shape, ratio, seed)
    test = ~train
    t0 = time.time()
    if model == "IRT-2PL":
        P = fit_irt2pl(R, train, seed)
    elif model == "DINA":
        P = fit_dina(R, Q, train, seed=seed)
    else:
        spec = spec or STANDART[model]
        w = observation_weights(R, D["delta_t"], D["duration"],
                                use_time=spec.use_time, use_duration=spec.use_duration)
        cfg = MCMCConfig(n_iter=spec.n_iter, burn=spec.burn, thin=5, seed=seed,
                         learn_mu=spec.learn_mu, and_kind=spec.and_kind, or_kind=spec.or_kind)
        P = run_mh(R, Q, subj, train, w, cfg)["P_mean"]
    sure = time.time() - t0
    sonuc = {"veri": data_name, "model": model, "oran": ratio, "seed": seed, "sure_sn": sure}
    sonuc.update({f"{k}": v for k, v in evaluate(R, P, test).items()})
    if subj.any():
        for ad, m in (("obj", ~subj), ("subj", subj)):
            mk = test & m[None, :]
            sonuc.update({f"{k}_{ad}": v for k, v in evaluate(R, P, mk).items()})
    return sonuc


def _job(args):
    return run_one(*args)


def run_grid(jobs, n_proc=10):
    """jobs: (veri, model, oran, seed[, spec]) listesi; paralel çalıştırır."""
    from multiprocessing import get_context
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    with get_context("spawn").Pool(n_proc) as pool:
        return list(pool.imap_unordered(_job, jobs, chunksize=1))
