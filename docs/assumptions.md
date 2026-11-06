# Varsayımlar ve Makaledeki Belirsizlikler

Referans makale: Zhang & Wang, Scientific Reports 15:11531 (2025). Aşağıdaki maddeler
makalede belirsiz ya da tutarsız olan yerlerdir ve gerçekleştirmede **bizim yorumumuzu**
gösterir. Gerekçeleri rapora taşınacaktır.

## Belirsiz / eksik noktalar
1. **Zaman ağırlığı (Eş. 2, 4):** Eş. 2 ile Eş. 4 aynıdır. Ağırlığın η'ya nasıl uygulandığı
   belirsizdir. *Varsayım:* zaman ağırlığı gözlemin olabilirlik (likelihood) ağırlığıdır.
   Alternatif (η ile çarpım) ablation'da denenebilir.
2. **Cevap süresi ağırlığı (Eş. 7):** Normalize değildir. *Varsayım:* ωC, maxC_n'e bölünerek
   [0,1]'e ölçeklenir; gözlem ağırlığı `w = ω_time · (1 + λ·ωC)`, λ = 1 (hiperparametre).
   Yanlış cevapta ωC = 0. Bu, makaleden değil bizim yorumumuzdur.
3. **Gösterim:** Eş. 1'de `a` hem ayırt edicilik hem yeterlilik için kullanılmış. Biz
   α_k (ayırt edicilik), β_k (zorluk), a_ik (yeterlilik) kullanıyoruz.
4. **Olabilirlik (Eş. 12–13):** İşaret/gösterim hataları var. Standart Bernoulli
   olabilirliği kullanılacak. Eş. 29–30'da P ve r tanımları ters.
5. **Skor türü:** Veri tasarımında skor "ikili" deniyor, fakat Eş. 7 kesirli puan oranı kullanıyor.
6. **Hiperparametreler:** Önerilen model için verilmemiş (yalnızca baseline'lar için:
   epoch 100, batch 16, lr 0.002).
7. **Tablo 4:** Metin "her tipte ve oranda en iyi" diyor, fakat subjektif %50 Recall'da
   C&RM-MAKT (0.532) MFCD-MAKT'tan (0.525) yüksek. PRE ve Recall'dan hesaplanan F1 de tablodakiyle
   birebir tutmuyor; makro/mikro ortalama belirtilmemiş.
8. **Öneri kalitesi:** Ölçülmemiş (ör. nDCG yok); yalnızca performans tahmini yapılmış.
9. **Kaynakça:** Güvenilir değil (ör. 25 numaralı kaynak 17'nin kopyası). Kopyalanmayacak,
   kaynaklar tek tek doğrulanarak eklenecek.

## Matematiksel not
μ_jk ikili (= q_jk) iken `η_ij = 1 − max_k q_jk(1 − a_ik) = min_{k: q=1} a_ik` olur; yani bilgi
açığı formülü Gödel t-norm'una (min) indirgenir. İkili Q-matrisli FrcSub'da açık faktörü min
operatöründen ayırt edilemez. Sentetik veride μ dereceli üretilecektir.

## Sentetik veri uyarısı
Sentetik veri zaman/süre etkisiyle üretildiğinden bu faktörleri içeren model doğal olarak
avantajlıdır (döngüsellik). Sonuçlar faktörlerin *geri kazanılabilirliğini* gösterir, gerçek
dünya faydasını kanıtlamaz.

## Sentetik veri üretecinin tasarım kararları (bize ait)
- Tek gizli yetenek θ_i; beceriler θ'dan sigmoid ile türetilir (FuzzyCDF ile aynı mantık).
- Zaman etkisi: öğrenci zamanla gelişir, `θ(Δt) = θ − 0.6·Δt/180`. Eski kayıtlar güncel
  duruma daha az bilgi taşır. Katsayı (0.6) keyfidir.
- Süre etkisi: yetkin öğrenci daha hızlıdır; şansla doğru bilenler çok hızlıdır. Katsayılar keyfidir.
- Subjektif sorular da ikili (0/1) üretilir; makaledeki kesirli puan oranı (Eş. 7) bu veride yoktur.
- Subjektif başarının objektiften düşük çıkması üretecin parametrelerinden gelir, bir bulgu değildir.
