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

## A5 / E1: FrcSub üzerinde karşılaştırma
Kaynak: `experiments/baseline/e1_frcsub.py` -> `results/tables/e1_frcsub_{raw,summary}.csv`,
`e1_frcsub_table.md`; grafik: `results/figures/e1_frcsub_f1_rmse.png`.
Kurulum: 4 model x 6 eğitim oranı (%40-90) x 10 seed (0-9) = 240 koşu; MH 1500 adım (700 burn-in).
Bölme: her öğrencinin kayıtlarının %p'si rastgele eğitim. Eşik 0.5, doğru cevap pozitif.

**Bulgular (ortalama; %60 eğitim):**
- F1: FuzzyCDF-orig 0.849 > MFCD-core 0.841 > IRT-2PL 0.834 > DINA 0.813.
- MAE'de IRT-2PL en iyi (0.241), RMSE'de FuzzyCDF-orig (0.352) ile MFCD-core (0.354) başa baş.
- Süre: IRT 0.04 sn, DINA 1.7 sn, FuzzyCDF-orig 8.6 sn, MFCD-core 10.1 sn (MH daha yavaş).
- Tüm modellerde eğitim oranı arttıkça F1 artıyor, RMSE azalıyor (makale eğilimi ile uyumlu).
- Makale eğilimi "DINA < FuzzyCDF" burada da var (F1 0.813 < 0.849). Bulanık modeller DINA'yı geçiyor.

**Beklenmeyen sonuç (olduğu gibi raporlanacak):** MFCD-core (μ öğrenilen) FrcSub'da
FuzzyCDF-orig'den **daha iyi değil**. Seed bazında eşleştirilmiş fark (F1, MFCD-core − FuzzyCDF-orig):
%50: -0.008 (10 seed'in 1'inde MFCD önde), %60: -0.008 (0/10), %70: -0.006 (0/10). Fark küçük ama tutarlı.
Olası yorumlar (test edilmemiş hipotezler): (i) FrcSub'da Q ikili ve veri yalnızca objektif; dereceli
μ ek serbestlik getirip 536 öğrencide aşırı uyuma yol açıyor olabilir; (ii) zaman/süre bilgisi yok,
yani üç faktörden hiçbiri bu veride kullanılamıyor; (iii) MH ile 20x8'lik μ öğrenmek zor.
Bu yorumlar K2 (ablation) ve sentetik veri deneyleriyle sınanacak.

**Ek notlar:**
- DINA kendi gerçekleştirmemiz (Gibbs örnekleyici); makaledeki DINA ile birebir aynı olmayabilir.
- IRT-2PL: MAP (L-BFGS), önseller zayıf; gradyan sayısal olarak doğrulandı (check_grad).
- Makale sayılarıyla birebir karşılaştırma yapılmaz (veri kapalı); yalnızca eğilimler.
- İlk E1 koşusunda IRT'de `exp` taşma uyarıları vardı (sonuç `clip` ile korunuyordu); `expit`'e
  geçilip E1 baştan koşuldu, nihai tablolar son koddan gelir.
