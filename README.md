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
