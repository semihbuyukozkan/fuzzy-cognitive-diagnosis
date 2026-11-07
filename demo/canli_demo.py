"""Canlı demo: tek bir öğrenci ve soru için bulanık çıkarım zincirini adım adım gösterir.

Zincir:  yetenek θ --(sigmoid üyelik)--> beceri dereceleri a_k --(VE / t-norm)--> η
         --(slip/guess)--> P(doğru) --(eşik 0.5 = "durulaştırma" karşılığı)--> doğru / yanlış

Model FrcSub üzerinde kısa bir MH koşusuyla (~5 sn) eğitilir (bu demo hızlı olsun diye
deneylerdeki 1500 yineleme yerine 600 yineleme kullanır; sayılar deney sonuçları DEĞİLDİR).

Kullanım örnekleri:
  python3 -m demo.canli_demo                          # öğrenci 0, soru 0
  python3 -m demo.canli_demo --ogrenci 12 --soru 5
  python3 -m demo.canli_demo --ogrenci 12 --soru 5 --theta 2.0     # "ya yeteneği yüksek olsaydı?"
  python3 -m demo.canli_demo --soru 3 --theta 0.3                   # çok becerili soruda 3 operatörün η'sı yan yana
  python3 -m demo.canli_demo --soru 5 --operator product            # modeli VE=çarpım ile baştan eğit
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from src.data_frcsub import load_frcsub
from src.data_split import split_per_student
from src.fuzzy_ops import and_reduce
from src.memberships import membership
from src.mcmc import MCMCConfig, run_mh
from src.mfcd import observation_weights


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ogrenci", type=int, default=0, help="öğrenci no (0-535)")
    ap.add_argument("--soru", type=int, default=0, help="soru no (0-19)")
    ap.add_argument("--theta", type=float, default=None, help="öğrencinin yeteneğini elle değiştir (what-if)")
    ap.add_argument("--operator", default="godel", choices=["godel", "product", "lukasiewicz"],
                    help="VE (t-norm) operatörü")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    R, Q = load_frcsub()
    subj = np.zeros(R.shape[1], bool)
    train = split_per_student(R.shape, 0.6, a.seed)
    print(f"Model eğitiliyor (FrcSub, %60 eğitim, VE={a.operator}, kısa koşu)...")
    cfg = MCMCConfig(n_iter=600, burn=300, thin=5, seed=a.seed, learn_mu=False, and_kind=a.operator)
    out = run_mh(R, Q, subj, train, observation_weights(R), cfg)
    p, i, j = out["params"], a.ogrenci, a.soru

    theta = out["theta_mean"][i] if a.theta is None else a.theta
    uyari = "" if a.theta is None else "  (elle verildi; modelin tahmini değil)"
    print(f"\nÖğrenci {i}, soru {j}: yetenek θ = {theta:+.2f}{uyari}")

    beceriler = np.where(Q[j] == 1)[0]
    A = membership(np.array([theta]), p.alpha, p.beta)[0]
    print("\n1) FUZZIFICATION: θ -> her beceri için üyelik derecesi (0 = hiç bilmiyor, 1 = tam biliyor)")
    for k in beceriler:
        print(f"   beceri {k}: a = 1/(1+exp(-1.7·{p.alpha[k]:.2f}·({theta:+.2f}-({p.beta[k]:+.2f})))) = {A[k]:.3f}")

    eta = float(and_reduce(A[beceriler][None, :], a.operator, axis=-1)[0])
    kural = " VE ".join(f"a{k}" for k in beceriler)
    print(f"\n2) KURAL / ÇIKARIM: \"{kural}\"  ({a.operator} t-norm; Gödel = min)")
    print(f"   η = {eta:.3f}   (soruyu çözmek için gereken tüm becerilere sahip olma derecesi)")

    if len(beceriler) > 1:                       # aynı üyeliklerle üç operatör yan yana (saf operatör farkı)
        diger = {k: float(and_reduce(A[beceriler][None, :], k, axis=-1)[0]) for k in ("godel", "product", "lukasiewicz")}
        print("   (aynı üyeliklerle) " + ", ".join(f"{k}: {v:.3f}" for k, v in diger.items())
              + "  -> Gödel ≥ çarpım ≥ Łukasiewicz")
    s, g = p.slip[j], p.guess[j]
    P = (1 - s) * eta + g * (1 - eta)
    print(f"\n3) SLIP/GUESS: P(doğru) = (1-{s:.2f})·{eta:.3f} + {g:.2f}·(1-{eta:.3f}) = {P:.3f}")

    karar = "DOĞRU" if P >= 0.5 else "YANLIŞ"
    print(f"\n4) DURULAŞTIRMA (eşik 0.5): tahmin = {karar}")
    if a.theta is None:
        gercek = "DOĞRU" if R[i, j] == 1 else "YANLIŞ"
        bolum = "eğitimde GÖRÜLDÜ" if train[i, j] else "TEST (model görmedi)"
        print(f"   gerçek cevap = {gercek}  [{bolum}]")


if __name__ == "__main__":
    main()
