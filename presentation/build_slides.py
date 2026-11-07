"""Sunum derleyici: presentation/slides_template.html -> presentation/presentation.pdf (Chrome headless).
Küçük tablolar results/tables/*.csv dosyalarından doldurulur (sunumdaki deney sayıları elle yazılmaz).
Çalıştırma:  python3 -m presentation.build_slides
"""
import subprocess
from pathlib import Path

import pandas as pd

from report.build_report import CHROME, pm, tablo

KOK = Path(__file__).resolve().parents[1]
T = KOK / "results" / "tables"


def e1_mini():
    s = pd.read_csv(T / "e1_frcsub_summary.csv")
    rows = []
    for m in ["IRT-2PL", "DINA", "FuzzyCDF-orig", "MFCD-core"]:
        r = s[(s.model == m) & (s.oran == 0.6)].iloc[0]
        rows.append([m, pm(r, "PRE"), pm(r, "Recall"), pm(r, "F1"), pm(r, "RMSE")])
    return tablo(["FrcSub, %60", "PRE", "Recall", "F1", "RMSE"], rows)


def k2_mini():
    s = pd.read_csv(T / "k2_summary.csv")
    rows = []
    for m in ["hiçbiri", "+süre (bizim yorum)", "hepsi"]:
        a = s[(s.veri == "synthetic") & (s.model == m)].iloc[0]
        b = s[(s.veri == "synthetic_null") & (s.model == m)].iloc[0]
        rows.append([m, f"{a.F1_ort:.3f}", f"{a.AUC_ort:.3f}", f"{a.ort_P_ort:.2f}",
                     f"{b.F1_ort:.3f}", f"{b.AUC_ort:.3f}", f"{b.ort_P_ort:.2f}"])
    return tablo(["Model", "F1", "AUC", "ort. tahmin", "F1 (kontrol)", "AUC (kontrol)", "ort. tahmin (kontrol)"], rows)


def e4_mini():
    s = pd.read_csv(T / "e4_sure_summary.csv")
    rows = []
    for m in ["IRT-2PL", "DINA", "FuzzyCDF-orig", "MFCD-core"]:
        f = s[(s.veri == "frcsub") & (s.model == m)]["mean"].iloc[0]
        y = s[(s.veri == "synthetic") & (s.model == m)]["mean"].iloc[0]
        rows.append([m, f"{f:.2f}", f"{y:.2f}"])
    return tablo(["Süre (sn, tek işlem)", "FrcSub", "Sentetik"], rows)


def main():
    h = (KOK / "presentation" / "slides_template.html").read_text(encoding="utf-8")
    for k, v in {"E1MINI": e1_mini(), "K2MINI": k2_mini(), "E4MINI": e4_mini()}.items():
        h = h.replace("{{" + k + "}}", v)
    assert "{{" not in h
    out = KOK / "presentation" / "slides.html"
    out.write_text(h, encoding="utf-8")
    pdf = KOK / "presentation" / "presentation.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out.as_uri()], check=True, capture_output=True)
    print("yazıldı:", pdf)


if __name__ == "__main__":
    main()
