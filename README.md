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
Testler: `python3 -m pytest -q`. Deney komutları [Reproduction Experiments](#reproduction-experiments) bölümünde.

## Reproduction Experiments
**E1: FrcSub üzerinde karşılaştırma** (10 seed, ortalama ± std; ayrıntılı yorum: [docs/report_notes.md](docs/report_notes.md))

```bash
python3 -m src.data_frcsub                 # veriyi indir ve doğrula
python3 -m experiments.baseline.e1_frcsub  # 240 koşu (~5 dk, 10 süreç)
python3 -m experiments.baseline.plot_e1    # grafik
python3 -m experiments.baseline.e1_table   # tablo
```

| Eğitim oranı | Model | PRE | Recall | F1 | MAE | RMSE | Süre (sn) |
|---|---|---|---|---|---|---|---|
| 50% | IRT-2PL | 0.814 ± 0.006 | 0.852 ± 0.007 | 0.832 ± 0.002 | 0.242 ± 0.001 | 0.367 ± 0.002 | 0.04 ± 0.01 |
| 50% | DINA | 0.844 ± 0.005 | 0.755 ± 0.004 | 0.797 ± 0.003 | 0.302 ± 0.002 | 0.383 ± 0.001 | 1.76 ± 0.08 |
| 50% | FuzzyCDF-orig | 0.820 ± 0.006 | 0.868 ± 0.005 | 0.843 ± 0.003 | 0.260 ± 0.001 | 0.357 ± 0.002 | 8.27 ± 0.07 |
| 50% | MFCD-core | 0.820 ± 0.007 | 0.852 ± 0.013 | 0.836 ± 0.008 | 0.265 ± 0.004 | 0.359 ± 0.002 | 10.03 ± 0.07 |
| 60% | IRT-2PL | 0.815 ± 0.007 | 0.855 ± 0.006 | 0.834 ± 0.005 | 0.241 ± 0.002 | 0.363 ± 0.004 | 0.04 ± 0.01 |
| 60% | DINA | 0.850 ± 0.006 | 0.780 ± 0.010 | 0.813 ± 0.005 | 0.284 ± 0.003 | 0.372 ± 0.003 | 1.70 ± 0.08 |
| 60% | FuzzyCDF-orig | 0.823 ± 0.005 | 0.877 ± 0.008 | 0.849 ± 0.005 | 0.253 ± 0.002 | 0.352 ± 0.003 | 8.57 ± 0.05 |
| 60% | MFCD-core | 0.824 ± 0.007 | 0.858 ± 0.014 | 0.841 ± 0.007 | 0.257 ± 0.004 | 0.354 ± 0.004 | 10.11 ± 0.07 |
| 70% | IRT-2PL | 0.816 ± 0.006 | 0.865 ± 0.007 | 0.840 ± 0.003 | 0.238 ± 0.002 | 0.357 ± 0.003 | 0.04 ± 0.01 |
| 70% | DINA | 0.845 ± 0.005 | 0.795 ± 0.009 | 0.819 ± 0.005 | 0.272 ± 0.002 | 0.367 ± 0.003 | 1.71 ± 0.04 |
| 70% | FuzzyCDF-orig | 0.826 ± 0.006 | 0.887 ± 0.007 | 0.856 ± 0.003 | 0.245 ± 0.002 | 0.346 ± 0.003 | 8.59 ± 0.05 |
| 70% | MFCD-core | 0.830 ± 0.009 | 0.870 ± 0.014 | 0.849 ± 0.006 | 0.248 ± 0.003 | 0.347 ± 0.004 | 10.16 ± 0.04 |

![E1](results/figures/e1_frcsub_f1_rmse.png)

Bulanık modeller (FuzzyCDF-orig, MFCD-core) DINA'yı geçiyor; bu makaledeki eğilimle uyumludur. Dereceli μ'nün
eklendiği MFCD-core ise FrcSub'da FuzzyCDF-orig'i geçmiyor (F1 farkı küçük ama tutarlı). Bu beklenmeyen sonuç olduğu gibi raporlanmıştır.

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
