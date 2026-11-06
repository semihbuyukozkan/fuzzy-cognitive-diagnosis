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
