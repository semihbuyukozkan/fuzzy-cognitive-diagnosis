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

T_M = 180.0

SEEDS = list(range(10))                       # sabit ve kayıtlı seed'ler
RATIOS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class MfcdSpec:
    """MFCD çalıştırma ayarı (K1 ve K2 için)."""
    learn_mu: bool = False        # açık faktörü (dereceli μ)
    use_time: bool = False        # zaman ağırlığı
    use_duration: bool = False    # süre ağırlığı
    time_mode: str = "likelihood" # zaman: "likelihood" (gözlem ağırlığı; bizim varsayım) | "eta" (Eş. 3 çarpımı)
    and_kind: str = "godel"
    or_kind: str = "godel"
    n_iter: int = 1500
    burn: int = 700
    n_restart: int = 1            # >1: farklı seed'lerle yeniden başlat, EĞİTİM olabilirliği en yüksek zinciri seç


STANDART = {
    "FuzzyCDF-orig": MfcdSpec(learn_mu=False),
    "MFCD-core": MfcdSpec(learn_mu=True),
}


def load_data(name, seed=0):
    """Dönüş: sözlük (R, Q, is_subj, delta_t, duration)."""
    if name == "frcsub":
        R, Q = load_frcsub()
        return dict(R=R, Q=Q, is_subj=np.zeros(R.shape[1], bool), delta_t=None, duration=None)
    d = generate(seed=seed, with_effects=(name != "synthetic_null"))   # veri seed'i = deney seed'i
    return dict(R=d.R, Q=d.Q, is_subj=d.is_subjective, delta_t=d.delta_t, duration=d.duration)


def _run_mfcd(R, Q, subj, D, train, spec, seed, **cfg_extra):
    """MFCD ayarına göre ağırlıkları/omega'yı kurar ve MH çalıştırır."""
    time_lik = spec.use_time and spec.time_mode == "likelihood"
    time_eta = spec.use_time and spec.time_mode == "eta"
    w = observation_weights(R, D["delta_t"], D["duration"], t_m=T_M,
                            use_time=time_lik, use_duration=spec.use_duration)
    omega = np.clip((T_M - D["delta_t"]) / T_M, 0.0, 1.0) if time_eta else None
    best, best_skor, skorlar = None, -np.inf, []
    for r in range(spec.n_restart):
        cfg = MCMCConfig(n_iter=spec.n_iter, burn=spec.burn, thin=5, seed=seed + 1000 * r,
                         learn_mu=spec.learn_mu, and_kind=spec.and_kind, or_kind=spec.or_kind,
                         **cfg_extra)
        out = run_mh(R, Q, subj, train, w, cfg, omega=omega)
        skor = float(out["ll_trace"][spec.burn:].mean())      # yalnızca eğitim verisi; test'e bakılmaz
        skorlar.append(skor)
        if skor > best_skor:
            best, best_skor = out, skor
    best["restart_ll_aralik"] = max(skorlar) - min(skorlar)    # büyükse bir başlangıç kötü çukura düştü
    return best


def run_chain(data_name, model, ratio, split_seed, chain_seed, n_iter=3000, burn=1500, spec=None,
              dispersed=True):
    """Yakınsama tanısı için tek zincir: aynı veri ve bölme, farklı seed.

    dispersed=True : dağınık başlangıç (R-hat için). False: üretim koşularındaki mantıklı başlangıç.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    D = load_data(data_name, split_seed)
    train = split_per_student(D["R"].shape, ratio, split_seed)
    base = spec or STANDART[model]
    sp = MfcdSpec(**{**base.__dict__, "n_iter": n_iter, "burn": burn})
    out = _run_mfcd(D["R"], D["Q"], D["is_subj"], D, train, sp, chain_seed,
                    overdispersed_init=dispersed, store_samples=True)
    met = evaluate(D["R"], out["P_mean"], ~train)
    return {"veri": data_name, "model": model, "zincir": chain_seed, "dispersed": dispersed,
            "F1": met["F1"], "RMSE": met["RMSE"], "ll_ort": float(out["ll_trace"][burn:].mean()),
            "samples": out["samples"], "ll_trace": out["ll_trace"], "ll_drift": out["ll_drift"],
            "burn": burn}


def _chain_job(args):
    return run_chain(*args)


def run_chains(jobs, n_proc=10):
    from multiprocessing import get_context
    os.environ["OMP_NUM_THREADS"] = "1"
    with get_context("spawn").Pool(n_proc) as pool:
        return pool.map(_chain_job, jobs, chunksize=1)


def run_one(data_name, model, ratio, seed, spec=None):
    """Tek koşu: bölme + eğitim + test metrikleri + süre. Dönüş: sözlük."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    D = load_data(data_name, seed)
    R, Q, subj = D["R"], D["Q"], D["is_subj"]
    train = split_per_student(R.shape, ratio, seed)
    test = ~train
    t0 = time.time()
    drift, aralik = float("nan"), float("nan")
    if model == "IRT-2PL":
        P = fit_irt2pl(R, train, seed)
    elif model == "DINA":
        P = fit_dina(R, Q, train, seed=seed)
    else:
        spec = spec or STANDART[model]
        out = _run_mfcd(R, Q, subj, D, train, spec, seed)
        P, drift, aralik = out["P_mean"], out["ll_drift"], out["restart_ll_aralik"]
    sure = time.time() - t0
    sonuc = {"veri": data_name, "model": model, "oran": ratio, "seed": seed, "sure_sn": sure,
              "ll_surukleme": drift, "restart_ll_aralik": aralik}
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
