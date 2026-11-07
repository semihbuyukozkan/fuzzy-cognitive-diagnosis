# Yapay Zekâ Kullanım Günlüğü

Raporun "Generative AI Usage Disclosure" bölümü bu dosyadan beslenecektir.

| Tarih | Araç | Amaç | Not |
|-------|------|------|-----|
| 2026-11-06 | Claude Code | Repo kurulumu, klasör iskeleti, dokümantasyon taslakları | Çıktılar öğrenci tarafından gözden geçirildi |
| (önceki) | Claude (sohbet) | Makale incelemesi, kapsam ve deney planı (`PROJECT_BRIEF.md`) | Planlama ve kavram açıklama |
| 2026-11-06 | Claude Code | A2: FrcSub indirme/doğrulama betiği, sentetik veri üreteci ve testleri | Veri boyutları gerçek indirme ile doğrulandı (536x20, K=8) |
| 2026-11-06 | Claude Code | A3: bulanık operatörler, üyelik fonksiyonu, MFCD çekirdeği ve testleri | Formüller makale denklemleriyle eşlendi; testler elle hesaplandı |
| 2026-11-06 | Claude Code | A4: Metropolis-Hastings kestirimi, geri kazanım betiği ve testleri | Adım boyları ve yakınsama sonuçları betik çıktısından alındı |
| 2026-11-06 | Claude Code | A5: metrikler, bölme, IRT-2PL/DINA baseline'ları, deney koşucusu, E1 ve grafik | Sayılar betik çıktısından; yorumlar öğrenci tarafından gözden geçirilecek |
| 2026-11-06 | Claude Code | A6: makale okuması, yakınsama tanıları (R-hat), 3 başlangıçlı E1, K1 ve K2 deneyleri, negatif kontrol, AUC/kalibrasyon analizi | Süre faktörünün F1 'kazancı' negatif kontrol ve AUC ile çürütüldü; sonuçlar betik çıktısından |

## A7: rapor iskeleti (2026-11-07)
- Claude Code ile yönergenin 11 bölümüne göre rapor (report/report_template.html) ve derleyici (report/build_report.py) yazıldı;
  tablolar CSV'lerden otomatik üretilir, PDF Chrome (headless) ile alınır. Kaynaklar web aramasıyla doğrulandı (references/README.md).
- Yazar kontrol listesi: rapordaki her cümlenin repodaki sonuçla uyumu, Generative AI Disclosure metni ve kaynakların okunması yazar tarafından yapılmalıdır.

## K1 döngüsellik kontrolü (2026-11-07)
- Claude Code ile `generate(operator="product")`, `experiments/proposed/k1_product_veri.py` ve test yazıldı; 50 koşu çalıştırıldı, sonuçlar rapora (Tablo 3a) ve notlara işlendi.

## Sunum ve savunma hazırlığı (2026-11-07)
- Claude Code ile sunum (presentation/slides_template.html, build_slides.py; 19 slayt, küçük tablolar CSV'den) ve olası soru-cevap listesi (docs/savunma_hazirlik.md) hazırlandı.
- Canlı demo, aynı üyeliklerle üç operatörü yan yana gösterecek şekilde genişletildi.
- Yazar yapacaklar: sunumu kendi sözleriyle prova etmek, soru-cevap listesini okuyup anlamak.
