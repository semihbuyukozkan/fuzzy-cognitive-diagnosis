"""E1 özet CSV'sinden Markdown tablosu üretir (50/60/70 eğitim oranları; ortalama ± std).

Çalıştırma:  python3 -m experiments.baseline.e1_table
Çıktı:       results/tables/e1_frcsub_table.md
"""
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[2]
MODELLER = ["IRT-2PL", "DINA", "FuzzyCDF-orig", "MFCD-core"]


def main():
    s = pd.read_csv(KOK / "results" / "tables" / "e1_frcsub_summary.csv")
    satir = ["| Eğitim oranı | Model | PRE | Recall | F1 | MAE | RMSE | Süre (sn) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in [0.5, 0.6, 0.7]:
        for m in MODELLER:
            x = s[(s.oran == r) & (s.model == m)].iloc[0]

            def f(k, n=3):
                return f"{x[k + '_ort']:.{n}f} ± {x[k + '_std']:.{n}f}"
            satir.append(f"| {int(r * 100)}% | {m} | {f('PRE')} | {f('Recall')} | {f('F1')} | "
                         f"{f('MAE')} | {f('RMSE')} | {f('sure_sn', 2)} |")
    (KOK / "results" / "tables" / "e1_frcsub_table.md").write_text("\n".join(satir) + "\n", encoding="utf-8")
    print("\n".join(satir))


if __name__ == "__main__":
    main()
