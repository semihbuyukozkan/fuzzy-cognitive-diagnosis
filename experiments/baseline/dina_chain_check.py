"""DINA (Gibbs) zincirleri arası tutarlılık: 4 farklı seed, aynı veri ve bölme (oran 0.6, bölme seed 0).

Çalıştırma:  python3 -m experiments.baseline.dina_chain_check
Çıktı:       results/tables/dina_chain_check.csv
"""
from pathlib import Path

import pandas as pd

from src.baselines import fit_dina
from src.data_split import split_per_student
from src.experiment import load_data
from src.metrics import evaluate

OUT = Path(__file__).resolve().parents[2] / "results" / "tables" / "dina_chain_check.csv"


def main():
    satir = []
    for veri in ("frcsub", "synthetic"):
        D = load_data(veri, 0)
        train = split_per_student(D["R"].shape, 0.6, 0)
        for c in (100, 101, 102, 103):
            P = fit_dina(D["R"], D["Q"], train, seed=c)
            m = evaluate(D["R"], P, ~train)
            satir.append({"veri": veri, "zincir": c, "F1": m["F1"], "RMSE": m["RMSE"]})
    df = pd.DataFrame(satir)
    df.to_csv(OUT, index=False)
    print(df.round(4).to_string(index=False))
    print(df.groupby("veri")[["F1", "RMSE"]].agg(["min", "max"]).round(4))


if __name__ == "__main__":
    main()
