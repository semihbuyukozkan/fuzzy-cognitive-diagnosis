"""Rapor derleyici: report/report_template.html içindeki {{TABLO}} yer tutucularını results/tables/*.csv
dosyalarından doldurur (rapordaki hiçbir deney sayısı elle yazılmaz), report.html üretir ve
Google Chrome (headless) ile report.pdf'e çevirir.

Çalıştırma:  python3 -m report.build_report
"""
import subprocess
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[1]
T = KOK / "results" / "tables"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def pm(r, c, d=3):
    return f"{r[c + '_ort']:.{d}f} ± {r[c + '_std']:.{d}f}"


def tablo(basliklar, satirlar, sinif=""):
    h = "".join(f"<th>{b}</th>" for b in basliklar)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in s) + "</tr>" for s in satirlar)
    return f'<table class="{sinif}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def e1():
    s = pd.read_csv(T / "e1_frcsub_summary.csv")
    sira = ["IRT-2PL", "DINA", "FuzzyCDF-orig", "MFCD-core"]
    rows = []
    for o in (0.5, 0.6, 0.7):
        for m in sira:
            r = s[(s.model == m) & (s.oran == o)].iloc[0]
            rows.append([f"%{int(o*100)}", m] + [pm(r, c) for c in ("PRE", "Recall", "F1", "MAE", "RMSE")])
    return tablo(["Eğitim", "Model", "PRE", "Recall", "F1", "MAE", "RMSE"], rows)


def k1():
    s = pd.read_csv(T / "k1_summary.csv")
    rows = []
    for _, r in s[(s.veri == "frcsub") & (s.oran == 0.6)].iterrows():
        rows.append(["FrcSub (%60)", r.model, pm(r, "F1"), pm(r, "RMSE")])
    for _, r in s[s.veri == "synthetic"].iterrows():
        rows.append(["Sentetik (%60)", r.model, pm(r, "F1"), pm(r, "RMSE")])
    return tablo(["Veri", "Ayar", "F1", "RMSE"], rows)


def k2(veri):
    s = pd.read_csv(T / "k2_summary.csv")
    sira = ["hiçbiri", "+bilgi açığı", "+zaman (olabilirlik)", "+zaman (Eş.3 çarpımı)", "+bilgi açığı +zaman",
            "+süre (bizim yorum)", "hepsi", "IRT-2PL", "DINA"]
    rows = []
    for m in sira:
        r = s[(s.veri == veri) & (s.model == m)].iloc[0]
        rows.append([m, pm(r, "F1"), pm(r, "AUC"), pm(r, "RMSE"), f"{r.ort_P_ort:.3f}", f"{r.ort_R_ort:.3f}"])
    return tablo(["Model", "F1", "AUC", "RMSE", "ort. tahmin", "gerçek oran"], rows)


def e4():
    s = pd.read_csv(T / "e4_sure_summary.csv")
    rows = []
    for m in s.model.unique():
        f = s[(s.veri == "frcsub") & (s.model == m)]
        y = s[(s.veri == "synthetic") & (s.model == m)]
        fmt = lambda d: "–" if d.empty else f"{d['mean'].iloc[0]:.2f} ({d['std'].iloc[0]:.2f})"
        rows.append([m, fmt(f), fmt(y)])
    return tablo(["Model", "FrcSub (sn)", "Sentetik (sn)"], rows)


def baseline_vs_gelistirilen():
    """Yönergenin 6.2 tablosu: baseline = FuzzyCDF-orig; geliştirilen = MFCD-core. FrcSub %60."""
    e = pd.read_csv(T / "e1_frcsub_summary.csv")
    t = pd.read_csv(T / "e4_sure_summary.csv")
    b = e[(e.model == "FuzzyCDF-orig") & (e.oran == 0.6)].iloc[0]
    g = e[(e.model == "MFCD-core") & (e.oran == 0.6)].iloc[0]
    tb = t[(t.veri == "frcsub") & (t.model == "FuzzyCDF-orig")]["mean"].iloc[0]
    tg = t[(t.veri == "frcsub") & (t.model == "MFCD-core")]["mean"].iloc[0]
    rows = []
    for ad, c, d in (("F1", "F1", 3), ("RMSE", "RMSE", 3), ("MAE", "MAE", 3)):
        rows.append([ad, f"{b[c+'_ort']:.{d}f}", f"{g[c+'_ort']:.{d}f}", f"{g[c+'_ort']-b[c+'_ort']:+.{d}f}"])
    rows.append(["Süre (sn, tek zincir)", f"{tb:.2f}", f"{tg:.2f}", f"{tg-tb:+.2f}"])
    return tablo(["Ölçüt (FrcSub, %60)", "Baseline: FuzzyCDF-orig", "Geliştirilen: MFCD-core", "Değişim"], rows)


def main():
    sablon = (KOK / "report" / "report_template.html").read_text(encoding="utf-8")
    dolgu = {"E1": e1(), "K1": k1(), "K2_ETKILI": k2("synthetic"), "K2_KONTROL": k2("synthetic_null"),
             "E4": e4(), "BASELINE": baseline_vs_gelistirilen()}
    for k, v in dolgu.items():
        sablon = sablon.replace("{{" + k + "}}", v)
    assert "{{" not in sablon, "doldurulmamış yer tutucu var"
    html = KOK / "report" / "report.html"
    html.write_text(sablon, encoding="utf-8")
    pdf = KOK / "report" / "report.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    print("yazıldı:", pdf)


if __name__ == "__main__":
    main()
