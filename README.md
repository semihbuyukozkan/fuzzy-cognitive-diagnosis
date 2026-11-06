# Fuzzy Cognitive Diagnosis (MFCD-core)

Bulanık Mantık dersi dönem projesi: bir makalenin bulanık bilişsel tanı çekirdeğinin yeniden gerçekleştirilmesi ve özgün katkı.

> **Durum:** Çalışma sürüyor. Bu README her aşamada güncellenmektedir.

## Student Information
- **Ad Soyad:** Musa Semih Büyüközkan
- **Öğrenci No:** 1306230099
- **Ders:** Bulanık Mantık (2026-2027 Güz)

## Original Paper
Y. Zhang and Y. Wang, "A personalized recommendation algorithm for English exercises incorporating fuzzy cognitive models and multiple attention mechanisms," *Scientific Reports*, vol. 15, art. 11531, 2025. DOI: [10.1038/s41598-025-96489-3](https://doi.org/10.1038/s41598-025-96489-3)

## Problem Description
Bir öğrencinin henüz çözmediği bir alıştırmadaki başarısını, geçmiş cevaplarından tahmin etmek. Öğrencinin her bilgi noktasındaki yeterliliği kesin (var/yok) değil, 0 ile 1 arasında bir **üyelik derecesi** olarak modellenir.

## Method
Makalenin MFCD (geliştirilmiş fuzzy cognitive diagnosis) bileşeni: sigmoid üyelik fonksiyonu, objektif sorularda bulanık VE (t-norm), subjektif sorularda bulanık VEYA (t-conorm), bilgi açığı, slip/guess parametreleri ve Metropolis-Hastings ile parametre kestirimi. Dikkat tabanlı MAKT bileşeni **bilinçli olarak kapsam dışıdır**.

## Fuzzy System Architecture
Her öğrenci-soru çifti için akış (ayrıntı: [src/mfcd.py](src/mfcd.py)):

```
θ_i, α_k, β_k --(sigmoid üyelik, Eş.1)--> a_ik  (öğrencinin beceri k'daki yeterlilik derecesi)
a_ik, μ_jk    --(beceri katkısı)--------> ã_ijk
ã_ijk         --(bulanık VE: objektif / VEYA: subjektif)--> η_ij
η_ij, s_j, g_j --(slip / guess)---------> P(r_ij = 1) = (1-s_j) η_ij + g_j (1-η_ij)
```
- **Bulanık küme / üyelik:** "beceri k'yı bilenler" kümesinde öğrencinin üyelik derecesi 0-1 arasındadır.
- **VE (t-norm):** objektif sorular tüm gerekli becerileri ister. Gödel: min, çarpım: a·b, Łukasiewicz: max(0, a+b-1).
- **VEYA (t-conorm):** subjektif sorularda bir beceri diğerini telafi edebilir. Gödel: max, olasılıksal toplam: a+b-ab, Łukasiewicz: min(1, a+b).
- **Bilgi açığı:** objektif soruda `ã = 1 - μ(1 - a)`; Gödel ile birleşince makaledeki Eş. 8-9 elde edilir.
- Zaman ve cevap süresi, gözlemlerin olabilirlik ağırlığı olarak kullanılır.

## Dataset
- **FrcSub** (gerçek veri, kesir çıkarma): indirme betiği ile alınır, repoya konmaz.
- **Sentetik veri**: makalenin yapısını taklit eder; açıkça sentetik olarak raporlanır.

Ayrıntı: [data/README.md](data/README.md)

## Installation
```bash
git clone https://github.com/semihbuyukozkan/fuzzy-cognitive-diagnosis.git
cd fuzzy-cognitive-diagnosis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements
Python 3.10, bağımlılıklar [requirements.txt](requirements.txt) dosyasında.

## How to Run
_Kodlar eklendikçe doldurulacak._

## Reproduction Experiments
_E1 (FrcSub), E2 (sentetik) ve E4 (süre) sonuçları eklenecek._

## Proposed Extension
- **K1:** Bulanık operatör karşılaştırması (Gödel, çarpım, Łukasiewicz).
- **K2:** Zaman / süre / bilgi açığı faktörlerinin ablation analizi.

## Experimental Results
_Tüm tablo ve grafikler repodaki betiklerin çıktısından gelecektir._

## Limitations
Makale verisi kapalıdır, bu yüzden sayılar birebir karşılaştırılamaz. Ayrıntılar için bkz. [docs/assumptions.md](docs/assumptions.md).

## References
Bkz. [references/README.md](references/README.md).

## License
[MIT](LICENSE)
