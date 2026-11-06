"""FrcSub veri setini indirir, açar, boyutlarını doğrular ve yükler.

FrcSub: kesir çıkarma testi. 536 öğrenci x 20 soru, 8 beceri (bilgi noktası).
Cevaplar ikili (1 = doğru, 0 = yanlış). Zaman / süre bilgisi YOKTUR.

Kaynak (veri sahipleri): Wu, Liu, Liu, Chen, Su, Chen, Hu,
"Cognitive Modelling for Predicting Examinee Performance", IJCAI 2015.
Veri adresi: http://staff.ustc.edu.cn/~qiliuql/data/math2015.rar

Not: Paketteki TermsOfUse.txt yalnızca "Math" verisi (Math1/Math2) içindir ve
bu projede o verilere dokunulmaz. Yalnızca FrcSub klasörü açılır.
Ham veri repoya konmaz (data/raw/ .gitignore'dadır); bu betikle indirilir.

Kullanım:  python -m src.data_frcsub
"""
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

URL = "http://staff.ustc.edu.cn/~qiliuql/data/math2015.rar"
# İlk indirmede hesaplanan SHA-256 (dosyanın değişmediğini doğrulamak için).
EXPECTED_SHA256 = "61ae2410a734011ea2c5f63b37a91b64ecb6c102376037d06997f475af8d85fb"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ARCHIVE = RAW_DIR / "math2015.rar"
FRCSUB_DIR = RAW_DIR / "math2015" / "FrcSub"

# Beklenen boyutlar (PROJECT_BRIEF §3.2). Farklı çıkarsa durulur.
N_STUDENTS, N_ITEMS, N_SKILLS = 536, 20, 8


def download(force=False):
    """RAR arşivini data/raw/ içine indirir ve SHA-256 değerini doğrular."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and not force:
        print(f"Arşiv zaten var: {ARCHIVE}")
    else:
        print(f"İndiriliyor: {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA-256 uyuşmuyor!\n beklenen: {EXPECTED_SHA256}\n bulunan : {digest}\n"
            "Dosya değişmiş olabilir; durup kontrol edin."
        )
    print("SHA-256 doğrulandı.")


def extract():
    """Arşivden yalnızca FrcSub klasörünü ve TermsOfUse.txt dosyasını açar.

    macOS'te hazır gelen `bsdtar` RAR açabilir. Yoksa: brew install libarchive
    """
    cmd = ["bsdtar", "-xf", str(ARCHIVE), "-C", str(RAW_DIR),
           "--include", "math2015/FrcSub/*", "--include", "math2015/TermsOfUse.txt"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("bsdtar bulunamadı. RAR arşivini elle açıp data/raw/ içine koyun.")


def validate_frcsub(R, Q):
    """Boyutları ve değer aralıklarını kontrol eder; sorun varsa hata verir."""
    if R.shape != (N_STUDENTS, N_ITEMS):
        raise ValueError(f"R boyutu {R.shape}, beklenen {(N_STUDENTS, N_ITEMS)}")
    if Q.shape != (N_ITEMS, N_SKILLS):
        raise ValueError(f"Q boyutu {Q.shape}, beklenen {(N_ITEMS, N_SKILLS)}")
    if not np.isin(R, [0, 1]).all():
        raise ValueError("R ikili (0/1) değil")
    if not np.isin(Q, [0, 1]).all():
        raise ValueError("Q ikili (0/1) değil")
    if (Q.sum(axis=1) == 0).any():
        raise ValueError("Hiçbir beceri gerektirmeyen bir soru var")


def load_frcsub(directory=FRCSUB_DIR):
    """R (öğrenci x soru) ve Q (soru x beceri) matrislerini döndürür."""
    R = np.loadtxt(Path(directory) / "data.txt", dtype=int)
    Q = np.loadtxt(Path(directory) / "q.txt", dtype=int)
    validate_frcsub(R, Q)
    return R, Q


def main():
    download()
    extract()
    R, Q = load_frcsub()
    print(f"R: {R.shape}, Q: {Q.shape}")
    print(f"Genel doğru oranı: {R.mean():.3f}")
    print(f"Soru başına gereken beceri sayısı (min/maks): "
          f"{Q.sum(axis=1).min()}/{Q.sum(axis=1).max()}")


if __name__ == "__main__":
    main()
