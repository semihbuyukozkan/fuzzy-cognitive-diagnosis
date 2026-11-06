# Rapor için Çalışma Notları

Sonda rapora ve sunuma taşınacak kararlar, gözlemler ve ayrıntılar. Sayılar buraya
yalnızca betik çıktısından yazılır.

## Tasarım kararları
- **Birleşik formülasyon (bizim):** `ã = 1 − μ(1 − a)` (VE katmanı) ve `ã = μ·a` (VEYA katmanı),
  ardından seçilen t-norm / t-conorm ile birleştirilir. Gödel + VE durumunda bu, makaledeki
  bilgi açığı formülü (Eş. 8-9) ile birebir aynıdır. Böylece operatör karşılaştırması (K1) ve
  açık faktörü (K2) tek bir çerçevede ele alınır.
- **Açık faktörü kapatmak** = μ'yü ikili Q'ya eşitlemek. İkili Q'da bilgi açığı min'e iner;
  FrcSub'da (ikili Q) açık faktörü ayırt edilemez (matematiksel not, assumptions.md).
- Zaman/süre ağırlıkları olabilirlik ağırlığıdır (makale belirsiz; varsayım).
- Eş. 7 normalizasyonu bize aittir: `ωC = (maxC_j − C_ij)/maxC_j · R_ij`, `w = ω_time(1 + λωC)`, λ = 1.
- Zaman ağırlığı Δt = T_m olan kayıtta tam 0 olur (formül gereği); alt sınır konmadı.

## Savunma için hatırlatmalar
- Üyelik fonksiyonu: `src/memberships.py`; operatörler: `src/fuzzy_ops.py`; birleşim: `src/mfcd.py`.
- Elle hesaplı testler: `tests/test_fuzzy_ops.py`, `tests/test_mfcd.py`.

## Teknik notlar
- Python 3.10.11 (sistem), paketler `requirements.txt` ile; venv kullanılmadı.
- Bir yan not: `.venv` Python 3.13 ile açılınca pinli numpy/scipy derlenemedi (3.13 için uygun
  paket yok). Rapora girmeyecek, ama temiz kurulum testinde (A8) Python 3.10 belirtilmeli.

## A4: Metropolis-Hastings (MCMC)
- Uygulama: `src/mcmc.py`. Random-walk MH; θ ve (s, g, μ) vektörel, α/β beceri beceri güncellenir.
- Önseller bizim seçimimiz: θ~N(0,1), β~N(0,1), log α~N(0,0.5²), s,g~U(0,0.4), μ~U(0,1].
- Zincir: 1500 adım, ilk 700 burn-in, her 10. örnek (tahmin = posterior ortalama P).
- Adım boyları kabul oranı ~0.25-0.55 olacak şekilde elle ayarlandı (θ 0.9, α/β 0.25, μ 0.15).
- **Yakınsama sorunu çıkmadı**, gradyan tabanlı MAP'e geçmek gerekmedi.
- Pilot sonuçlar (sentetik, seed 0, %60 eğitim; kaynak: `experiments/proposed/a4_mh_recovery.py`,
  `results/tables/a4_mh_recovery.csv`):
  θ korelasyonu 0.863 (toplam skor 0.837); test doğruluğu 0.664 (gerçek parametreler 0.674,
  çoğunluk sınıfı 0.532); log-olabilirlik 300-500 ve son 800 adımda aynı düzeyde (-17947 / -17949).
- Yorum: Test doğruluğu gerçek parametreli "oracle" tahmine yakın; kalan fark ikili cevaplardaki
  slip/guess gürültüsünden gelir. Tek seed; makalede istenen ≥10 seed deneyi A5-A6'da yapılacak.
- Süre: ~65 sn / zincir (1500 adım). Deney ızgarası bunu hesaba katarak kurulacak.
- Not: üretecin zaman kayması (drift) modele verilmediği için ağırlıksız model bunu görmez;
  zaman ağırlığının etkisi K2 ablation'ında ölçülecek.
