"""A4: Sentetik veride MH kestiriminin parametre geri kazanımı ve yakınsama kontrolü.

Çalıştırma:  python3 -m experiments.proposed.a4_mh_recovery
Çıktı:       results/tables/a4_mh_recovery.csv  (+ ekrana özet)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_synthetic import generate
from src.mcmc import MCMCConfig, run_mh
from src.mfcd import Params, predict_proba

SEED, TRAIN_RATIO = 0, 0.6
OUT = Path(__file__).resolve().parents[2] / "results" / "tables" / "a4_mh_recovery.csv"


def main():
    d = generate(seed=SEED)
    rng = np.random.default_rng(SEED)
    train = rng.random(d.R.shape) < TRAIN_RATIO
    test = ~train
    w = np.ones(d.R.shape)

    t0 = time.time()
    out = run_mh(d.R, d.Q, d.is_subjective, train, w, MCMCConfig(seed=SEED))
    sure = time.time() - t0

    P_gercek = predict_proba(Params(d.theta, d.alpha, d.beta, d.slip, d.guess),
                             d.Q, d.mu, d.is_subjective)
    skor = (d.R * train).sum(1) / train.sum(1)
    tr = out["ll_trace"]
    sonuc = {
        "seed": SEED, "egitim_orani": TRAIN_RATIO, "sure_sn": round(sure, 1),
        "theta_korelasyonu_MH": round(np.corrcoef(out["theta_mean"], d.theta)[0, 1], 3),
        "theta_korelasyonu_toplam_skor": round(np.corrcoef(skor, d.theta)[0, 1], 3),
        "test_dogruluk_MH": round(((out["P_mean"][test] > 0.5) == d.R[test]).mean(), 3),
        "test_dogruluk_gercek_parametre": round(((P_gercek[test] > 0.5) == d.R[test]).mean(), 3),
        "test_dogruluk_cogunluk": round(max(d.R[test].mean(), 1 - d.R[test].mean()), 3),
        "ll_300_500": round(tr[300:500].mean(), 1), "ll_son_800": round(tr[-800:].mean(), 1),
        **{f"kabul_{k}": round(v, 2) for k, v in out["accept"].items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([sonuc]).to_csv(OUT, index=False)
    for k, v in sonuc.items():
        print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
